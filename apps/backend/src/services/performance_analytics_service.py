#!/usr/bin/env python3
"""
Performance Analytics Service
Calculates comprehensive performance metrics for learning system
Tracks improvements over time and provides insights
"""

import logging
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import asyncpg
import numpy as np

logger = logging.getLogger(__name__)

class PerformanceAnalyticsService:
    """
    Calculates and tracks performance metrics for the learning system
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        
        # Cache for frequently accessed data
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.last_cache_update = {}
        
        logger.info("📊 Performance Analytics Service initialized")
    
    async def calculate_overall_performance(self, period='7d') -> Dict[str, Any]:
        """
        Calculate overall system performance metrics
        
        Returns:
            Dictionary containing:
            - total_signals: Total signals generated
            - win_rate: Percentage of winning signals
            - avg_profit_per_trade: Average profit percentage
            - profit_factor: Ratio of total wins / total losses
            - sharpe_ratio: Risk-adjusted returns
            - max_drawdown: Maximum consecutive loss
            - best_streak: Best consecutive wins
            - worst_streak: Worst consecutive losses
        """
        try:
            days = int(period.replace('d', ''))
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            async with self.db_pool.acquire() as conn:
                # Get all completed signals in period
                signals = await conn.fetch("""
                    SELECT 
                        signal_id, outcome, profit_loss_pct,
                        confidence, quality_score, signal_timestamp
                    FROM signal_history
                    WHERE signal_timestamp >= $1
                    AND outcome IN ('win', 'loss')
                    AND source = 'live'
                    ORDER BY signal_timestamp ASC
                """, cutoff_date)
                
                if not signals:
                    return {
                        'total_signals': 0,
                        'win_rate': 0.0,
                        'message': 'No data available for analysis'
                    }
                
                # Convert to list of dicts
                signals_list = [dict(s) for s in signals]
                
                # Calculate metrics
                total_signals = len(signals_list)
                wins = [s for s in signals_list if s['outcome'] == 'win']
                losses = [s for s in signals_list if s['outcome'] == 'loss']
                
                win_count = len(wins)
                loss_count = len(losses)
                win_rate = (win_count / total_signals) if total_signals > 0 else 0.0
                
                # Average profit per trade
                profits = [float(s['profit_loss_pct']) for s in signals_list]
                avg_profit = np.mean(profits) if profits else 0.0
                
                # Profit factor (total wins / total losses)
                total_win_profit = sum([float(s['profit_loss_pct']) for s in wins])
                total_loss_profit = abs(sum([float(s['profit_loss_pct']) for s in losses]))
                profit_factor = (total_win_profit / total_loss_profit) if total_loss_profit > 0 else 0.0
                
                # Sharpe ratio (simplified)
                returns_std = np.std(profits) if len(profits) > 1 else 1.0
                sharpe_ratio = (avg_profit / returns_std) if returns_std > 0 else 0.0
                
                # Max drawdown
                max_drawdown = self._calculate_max_drawdown(signals_list)
                
                # Consecutive streaks
                best_streak, worst_streak = self._calculate_streaks(signals_list)
                
                return {
                    'total_signals': total_signals,
                    'win_count': win_count,
                    'loss_count': loss_count,
                    'win_rate': round(win_rate, 4),
                    'avg_profit_per_trade': round(avg_profit, 2),
                    'total_profit': round(sum(profits), 2),
                    'profit_factor': round(profit_factor, 2),
                    'sharpe_ratio': round(sharpe_ratio, 2),
                    'max_drawdown': round(max_drawdown, 2),
                    'best_streak': best_streak,
                    'worst_streak': worst_streak,
                    'period': period
                }
                
        except Exception as e:
            logger.error(f"❌ Error calculating overall performance: {e}")
            return {'error': str(e)}
    
    async def calculate_head_performance(self) -> Dict[str, Any]:
        """
        Calculate performance metrics for each of the 9 heads
        
        Returns:
            Dictionary with per-head statistics:
            - win_rate_when_agreed: Win rate for signals where head agreed
            - signals_contributed: Number of signals head participated in
            - current_weight: Current head weight
            - suggested_weight: Optimal weight based on performance
            - performance_trend: 'improving', 'stable', 'declining'
        """
        try:
            async with self.db_pool.acquire() as conn:
                # Get recent signals with SDE consensus data
                signals = await conn.fetch("""
                    SELECT 
                        signal_id, outcome, profit_loss_pct,
                        sde_consensus, signal_timestamp
                    FROM signal_history
                    WHERE signal_timestamp >= NOW() - INTERVAL '30 days'
                    AND outcome IN ('win', 'loss')
                    AND sde_consensus IS NOT NULL
                    AND source = 'live'
                    ORDER BY signal_timestamp DESC
                """)
                
                if not signals:
                    return {'message': 'No data available for head performance analysis'}
                
                # Get current head weights
                current_weights = await self._get_current_head_weights(conn)
                
                # Analyze each head
                head_stats = {}
                head_names = ['HEAD_A', 'HEAD_B', 'HEAD_C', 'HEAD_D', 'HEAD_E', 
                            'HEAD_F', 'HEAD_G', 'HEAD_H', 'HEAD_I']
                
                for head_name in head_names:
                    head_stats[head_name] = await self._analyze_head_performance(
                        head_name, signals, current_weights
                    )
                
                return {
                    'heads': head_stats,
                    'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
                    'signals_analyzed': len(signals)
                }
                
        except Exception as e:
            logger.error(f"❌ Error calculating head performance: {e}")
            return {'error': str(e)}
    
    async def _analyze_head_performance(
        self, 
        head_name: str, 
        signals: List, 
        current_weights: Dict
    ) -> Dict[str, Any]:
        """
        Analyze performance of a specific head
        """
        try:
            signals_with_head = []
            wins_with_head = 0
            
            for signal in signals:
                sde_consensus = signal['sde_consensus']
                if isinstance(sde_consensus, str):
                    sde_consensus = json.loads(sde_consensus)
                
                heads = sde_consensus.get('heads', {})
                if head_name in heads:
                    head_data = heads[head_name]
                    if isinstance(head_data, dict):
                        # Check if head agreed (not FLAT)
                        direction = head_data.get('direction', 'FLAT')
                        if direction in ['LONG', 'SHORT']:
                            signals_with_head.append(signal)
                            if signal['outcome'] == 'win':
                                wins_with_head += 1
            
            # Calculate metrics
            signals_count = len(signals_with_head)
            win_rate = (wins_with_head / signals_count) if signals_count > 0 else 0.0
            
            # Current weight
            current_weight = current_weights.get(head_name, 0.111)
            
            # Suggested weight based on performance
            # If win rate > 0.65, suggest increase; if < 0.55, suggest decrease
            if win_rate > 0.65:
                suggested_weight = min(0.30, current_weight * 1.1)
            elif win_rate < 0.55:
                suggested_weight = max(0.05, current_weight * 0.9)
            else:
                suggested_weight = current_weight
            
            # Performance trend (compare recent vs older signals)
            trend = self._calculate_performance_trend(signals_with_head)
            
            return {
                'win_rate_when_agreed': round(win_rate, 4),
                'signals_contributed': signals_count,
                'wins': wins_with_head,
                'losses': signals_count - wins_with_head,
                'current_weight': round(current_weight, 4),
                'suggested_weight': round(suggested_weight, 4),
                'weight_adjustment_needed': round(suggested_weight - current_weight, 4),
                'performance_trend': trend
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {head_name}: {e}")
            return {
                'error': str(e),
                'current_weight': current_weights.get(head_name, 0.111)
            }
    
    def _calculate_performance_trend(self, signals: List) -> str:
        """
        Calculate if performance is improving, stable, or declining
        """
        if len(signals) < 10:
            return 'insufficient_data'
        
        # Split into recent and older
        mid_point = len(signals) // 2
        recent = signals[:mid_point]
        older = signals[mid_point:]
        
        recent_win_rate = sum(1 for s in recent if s['outcome'] == 'win') / len(recent)
        older_win_rate = sum(1 for s in older if s['outcome'] == 'win') / len(older)
        
        diff = recent_win_rate - older_win_rate
        
        if diff > 0.05:
            return 'improving'
        elif diff < -0.05:
            return 'declining'
        else:
            return 'stable'
    
    async def calculate_learning_progress(self) -> Dict[str, Any]:
        """
        Track learning progress over time
        Shows week-over-week improvements
        """
        try:
            async with self.db_pool.acquire() as conn:
                # Get performance for each week
                weeks_data = []
                
                for week in range(4):  # Last 4 weeks
                    start_date = datetime.now(timezone.utc) - timedelta(days=(week + 1) * 7)
                    end_date = datetime.now(timezone.utc) - timedelta(days=week * 7)
                    
                    signals = await conn.fetch("""
                        SELECT outcome, profit_loss_pct
                        FROM signal_history
                        WHERE signal_timestamp >= $1 AND signal_timestamp < $2
                        AND outcome IN ('win', 'loss')
                        AND source = 'live'
                    """, start_date, end_date)
                    
                    if signals:
                        total = len(signals)
                        wins = sum(1 for s in signals if s['outcome'] == 'win')
                        win_rate = wins / total if total > 0 else 0
                        avg_profit = np.mean([float(s['profit_loss_pct']) for s in signals])
                        
                        weeks_data.append({
                            'week': f'Week {4 - week}',
                            'start_date': start_date.date().isoformat(),
                            'end_date': end_date.date().isoformat(),
                            'signals': total,
                            'win_rate': round(win_rate, 4),
                            'avg_profit': round(avg_profit, 2)
                        })
                
                # Calculate improvement trend
                if len(weeks_data) >= 2:
                    earliest_win_rate = weeks_data[-1]['win_rate']
                    latest_win_rate = weeks_data[0]['win_rate']
                    improvement = latest_win_rate - earliest_win_rate
                    
                    return {
                        'weekly_data': weeks_data,
                        'overall_improvement': round(improvement, 4),
                        'trend': 'improving' if improvement > 0.02 else 'stable' if improvement > -0.02 else 'declining'
                    }
                else:
                    return {
                        'weekly_data': weeks_data,
                        'message': 'Need more data for trend analysis'
                    }
                    
        except Exception as e:
            logger.error(f"❌ Error calculating learning progress: {e}")
            return {'error': str(e)}
    
    def _calculate_max_drawdown(self, signals: List[Dict]) -> float:
        """
        Calculate maximum drawdown (largest cumulative loss)
        """
        if not signals:
            return 0.0
        
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        
        for signal in signals:
            profit = float(signal['profit_loss_pct'])
            cumulative += profit
            
            if cumulative > peak:
                peak = cumulative
            
            drawdown = peak - cumulative
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    def _calculate_streaks(self, signals: List[Dict]) -> tuple:
        """
        Calculate best and worst consecutive win/loss streaks
        """
        if not signals:
            return (0, 0)
        
        current_streak = 0
        best_streak = 0
        worst_streak = 0
        last_outcome = None
        
        for signal in signals:
            outcome = signal['outcome']
            
            if outcome == last_outcome:
                if outcome == 'win':
                    current_streak += 1
                else:
                    current_streak -= 1
            else:
                # Streak broken
                if current_streak > best_streak:
                    best_streak = current_streak
                if current_streak < worst_streak:
                    worst_streak = current_streak
                
                # Start new streak
                current_streak = 1 if outcome == 'win' else -1
                last_outcome = outcome
        
        # Check final streak
        if current_streak > best_streak:
            best_streak = current_streak
        if current_streak < worst_streak:
            worst_streak = current_streak
        
        return (best_streak, abs(worst_streak))
    
    async def _get_current_head_weights(self, conn) -> Dict[str, float]:
        """
        Get current head weights from database
        """
        try:
            result = await conn.fetchrow("""
                SELECT state_data
                FROM active_learning_state
                WHERE state_type = 'head_weights'
            """)
            
            if result:
                weights = result['state_data']
                if isinstance(weights, str):
                    weights = json.loads(weights)
                return weights
            else:
                # Default weights
                return {f'HEAD_{chr(65+i)}': 0.111 for i in range(9)}
                
        except Exception as e:
            logger.warning(f"⚠️ Error getting head weights: {e}")
            return {f'HEAD_{chr(65+i)}': 0.111 for i in range(9)}
    
    async def get_weight_history(self, state_type: str, days: int = 30) -> List[Dict]:
        """
        Get historical weight changes over time
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            async with self.db_pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT 
                        version, state_data, performance_metrics,
                        created_at, created_by
                    FROM learning_state
                    WHERE state_type = $1
                    AND created_at >= $2
                    ORDER BY version ASC
                """, state_type, cutoff_date)
                
                history = []
                for row in results:
                    state_data = row['state_data']
                    if isinstance(state_data, str):
                        state_data = json.loads(state_data)
                    
                    perf_metrics = row['performance_metrics']
                    if isinstance(perf_metrics, str):
                        perf_metrics = json.loads(perf_metrics)
                    
                    history.append({
                        'version': row['version'],
                        'weights': state_data,
                        'performance': perf_metrics,
                        'created_at': row['created_at'].isoformat(),
                        'created_by': row['created_by']
                    })
                
                return history
                
        except Exception as e:
            logger.error(f"❌ Error getting weight history: {e}")
            return []

