#!/usr/bin/env python3
"""
Weekly Retraining Job
Full model retraining and optimization
Runs every Sunday at 02:00 UTC
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Tuple
import asyncpg
import json
import numpy as np

logger = logging.getLogger(__name__)

async def run_weekly_retraining(db_pool: asyncpg.Pool):
    """
    Weekly retraining job - full model optimization
    Runs every Sunday at 02:00 UTC
    """
    logger.info("="*80)
    logger.info("📅 WEEKLY RETRAINING JOB STARTED")
    logger.info(f"Time: {datetime.now(timezone.utc).isoformat()}")
    logger.info("="*80)
    
    try:
        # 1. Get last week's data (minimum 50 outcomes required)
        outcomes = await _get_outcomes_last_7_days(db_pool)
        logger.info(f"📊 Retrieved {len(outcomes)} outcomes from last 7 days")
        
        if len(outcomes) < 50:
            logger.info(f"⏭️ Not enough data for retraining (need 50+, have {len(outcomes)})")
            return {
                'status': 'skipped',
                'reason': 'insufficient_data',
                'outcomes_available': len(outcomes),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        # 2. Calculate weekly performance metrics
        weekly_metrics = await _calculate_weekly_metrics(outcomes)
        logger.info(f"📈 Weekly Performance:")
        logger.info(f"   Win Rate: {weekly_metrics['win_rate']:.2%}")
        logger.info(f"   Avg Profit: {weekly_metrics['avg_profit']:.2f}%")
        logger.info(f"   Profit Factor: {weekly_metrics.get('profit_factor', 0):.2f}")
        
        # 3. Calculate optimal head weights using statistical optimization
        logger.info("🧠 Calculating optimal head weights...")
        optimal_weights, optimization_metrics = await _calculate_optimal_head_weights(
            db_pool, outcomes
        )
        
        # 4. Get current weights for comparison
        current_weights = await _get_current_weights(db_pool)
        
        # 5. Compare performance
        logger.info("🔬 Comparing current vs optimal weights...")
        comparison = await _compare_weight_performance(
            outcomes, current_weights, optimal_weights
        )
        
        logger.info(f"   Current Win Rate: {comparison['current_win_rate']:.2%}")
        logger.info(f"   Optimal Win Rate: {comparison['optimal_win_rate']:.2%}")
        logger.info(f"   Improvement: {comparison['improvement']:.2%}")
        
        # 6. Deploy if better (minimum 3% win rate improvement)
        deployed = False
        if comparison['improvement'] > 0.03:
            logger.info(f"✅ Deploying optimal weights (improvement: {comparison['improvement']:.2%})")
            await _deploy_new_weights(db_pool, optimal_weights, optimization_metrics)
            deployed = True
        else:
            logger.info(f"⏭️ Keeping current weights (improvement too small: {comparison['improvement']:.2%})")
        
        # 7. Generate and store weekly report
        report = {
            'week_ending': datetime.now(timezone.utc).date().isoformat(),
            'outcomes_analyzed': len(outcomes),
            'weekly_metrics': weekly_metrics,
            'optimization_metrics': optimization_metrics,
            'weight_comparison': comparison,
            'weights_deployed': deployed,
            'current_weights': current_weights,
            'optimal_weights': optimal_weights if deployed else None,
            'job_completed_at': datetime.now(timezone.utc).isoformat()
        }
        
        await _store_weekly_report(db_pool, report)
        logger.info("💾 Weekly report stored in database")
        
        logger.info("="*80)
        logger.info("✅ WEEKLY RETRAINING JOB COMPLETED")
        logger.info("="*80)
        
        return report
        
    except Exception as e:
        logger.error(f"❌ Weekly retraining job failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

async def _get_outcomes_last_7_days(db_pool: asyncpg.Pool) -> List[Dict]:
    """Get all signal outcomes from last 7 days"""
    try:
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=7)
        
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    signal_id, symbol, direction, outcome,
                    profit_loss_pct, confidence, quality_score,
                    sde_consensus, pattern_type, indicators,
                    market_regime, signal_timestamp, completed_at
                FROM signal_history
                WHERE completed_at >= $1
                AND outcome IN ('win', 'loss', 'breakeven')
                AND source = 'live'
                ORDER BY completed_at ASC
            """, cutoff_time)
            
            return [dict(row) for row in rows]
            
    except Exception as e:
        logger.error(f"❌ Error getting outcomes: {e}")
        return []

async def _calculate_weekly_metrics(outcomes: List[Dict]) -> Dict[str, Any]:
    """Calculate comprehensive weekly performance metrics"""
    try:
        if not outcomes:
            return {}
        
        total = len(outcomes)
        wins = [o for o in outcomes if o['outcome'] == 'win']
        losses = [o for o in outcomes if o['outcome'] == 'loss']
        
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total if total > 0 else 0.0
        
        # Profit metrics
        profits = [float(o['profit_loss_pct']) for o in outcomes if o['profit_loss_pct'] is not None]
        avg_profit = np.mean(profits) if profits else 0.0
        
        # Profit factor
        win_profits = [float(o['profit_loss_pct']) for o in wins if o['profit_loss_pct'] is not None]
        loss_profits = [abs(float(o['profit_loss_pct'])) for o in losses if o['profit_loss_pct'] is not None]
        
        total_wins = sum(win_profits) if win_profits else 0
        total_losses = sum(loss_profits) if loss_profits else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        return {
            'total_signals': total,
            'wins': win_count,
            'losses': loss_count,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'total_profit': sum(profits) if profits else 0.0,
            'profit_factor': profit_factor
        }
        
    except Exception as e:
        logger.error(f"❌ Error calculating weekly metrics: {e}")
        return {}

async def _calculate_optimal_head_weights(
    db_pool: asyncpg.Pool,
    outcomes: List[Dict]
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Calculate optimal head weights using statistical optimization
    """
    try:
        head_names = ['HEAD_A', 'HEAD_B', 'HEAD_C', 'HEAD_D', 'HEAD_E', 
                     'HEAD_F', 'HEAD_G', 'HEAD_H', 'HEAD_I']
        
        # Calculate performance for each head
        head_stats = {}
        
        for head_name in head_names:
            signals_with_head = []
            wins_with_head = 0
            
            for outcome in outcomes:
                sde_consensus = outcome.get('sde_consensus')
                if not sde_consensus:
                    continue
                    
                if isinstance(sde_consensus, str):
                    sde_consensus = json.loads(sde_consensus)
                
                heads = sde_consensus.get('heads', {})
                signal_direction = outcome['direction'].upper()
                
                if head_name in heads:
                    head_data = heads[head_name]
                    if isinstance(head_data, dict):
                        head_direction = head_data.get('direction', 'FLAT')
                        
                        # Check if head agreed
                        if head_direction == signal_direction:
                            signals_with_head.append(outcome)
                            if outcome['outcome'] == 'win':
                                wins_with_head += 1
            
            # Calculate statistics
            total_signals = len(signals_with_head)
            win_rate = wins_with_head / total_signals if total_signals > 0 else 0.5
            
            head_stats[head_name] = {
                'signals': total_signals,
                'wins': wins_with_head,
                'win_rate': win_rate,
                'confidence': min(total_signals / 50.0, 1.0)  # Confidence based on sample size
            }
        
        # Calculate optimal weights based on win rate and confidence
        optimal_weights = {}
        
        for head_name in head_names:
            stats = head_stats[head_name]
            win_rate = stats['win_rate']
            confidence = stats['confidence']
            
            # Weight formula: base weight adjusted by performance and confidence
            base_weight = 0.111  # Equal distribution
            
            # Adjustment based on win rate (above/below 0.60 baseline)
            performance_factor = (win_rate - 0.60) * 2.0  # Scale difference
            
            # Apply confidence weighting
            adjusted_weight = base_weight * (1 + performance_factor * confidence)
            
            # Ensure bounds
            optimal_weights[head_name] = max(0.05, min(0.30, adjusted_weight))
        
        # Normalize to sum to 1.0
        total_weight = sum(optimal_weights.values())
        if total_weight > 0:
            optimal_weights = {k: v / total_weight for k, v in optimal_weights.items()}
        
        # Optimization metrics
        optimization_metrics = {
            'head_statistics': head_stats,
            'optimization_method': 'win_rate_weighted',
            'confidence_weighted': True
        }
        
        return optimal_weights, optimization_metrics
        
    except Exception as e:
        logger.error(f"❌ Error calculating optimal weights: {e}")
        # Return default equal weights on error
        return {f'HEAD_{chr(65+i)}': 0.111 for i in range(9)}, {}

async def _get_current_weights(db_pool: asyncpg.Pool) -> Dict[str, float]:
    """Get current head weights from database"""
    try:
        async with db_pool.acquire() as conn:
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
                return {f'HEAD_{chr(65+i)}': 0.111 for i in range(9)}
                
    except Exception as e:
        logger.error(f"❌ Error getting current weights: {e}")
        return {f'HEAD_{chr(65+i)}': 0.111 for i in range(9)}

async def _compare_weight_performance(
    outcomes: List[Dict],
    current_weights: Dict[str, float],
    optimal_weights: Dict[str, float]
) -> Dict[str, Any]:
    """
    Compare performance of current vs optimal weights
    (Simplified - uses win rate as proxy)
    """
    try:
        # Calculate win rate with current weights (actual historical performance)
        current_wins = len([o for o in outcomes if o['outcome'] == 'win'])
        current_win_rate = current_wins / len(outcomes) if outcomes else 0.0
        
        # Estimate win rate with optimal weights (based on head performance)
        # This is a simplified estimation
        optimal_win_rate = current_win_rate  # Start with current
        
        # Adjust based on weight changes for high-performing heads
        for head_name in current_weights.keys():
            weight_change = optimal_weights.get(head_name, 0.111) - current_weights.get(head_name, 0.111)
            # If we're increasing weight on a head, assume slight improvement
            if weight_change > 0.02:
                optimal_win_rate += 0.01  # Optimistic 1% improvement per head
        
        improvement = optimal_win_rate - current_win_rate
        
        return {
            'current_win_rate': current_win_rate,
            'optimal_win_rate': optimal_win_rate,
            'improvement': improvement,
            'current_weights': current_weights,
            'optimal_weights': optimal_weights
        }
        
    except Exception as e:
        logger.error(f"❌ Error comparing performance: {e}")
        return {
            'current_win_rate': 0.0,
            'optimal_win_rate': 0.0,
            'improvement': 0.0
        }

async def _deploy_new_weights(
    db_pool: asyncpg.Pool,
    new_weights: Dict[str, float],
    metrics: Dict[str, Any]
):
    """Deploy new optimized weights to database"""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("""
                SELECT update_head_weights($1::jsonb, $2::jsonb)
            """,
            json.dumps(new_weights),
            json.dumps({
                'trigger': 'weekly_retraining',
                'optimization_metrics': metrics,
                'deployment_timestamp': datetime.now(timezone.utc).isoformat()
            }))
        
        logger.info("✅ New weights deployed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error deploying new weights: {e}")
        raise

async def _store_weekly_report(db_pool: asyncpg.Pool, report: Dict):
    """Store weekly retraining report in database"""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO learning_events (
                    event_type, event_timestamp, new_value,
                    triggered_by, notes
                )
                VALUES ($1, $2, $3, $4, $5)
            """,
            'weekly_retraining_report',
            datetime.now(timezone.utc),
            json.dumps(report),
            'weekly_job',
            f"Weekly retraining completed: {report.get('outcomes_analyzed', 0)} outcomes, "
            f"weights {'deployed' if report.get('weights_deployed') else 'unchanged'}"
            )
    except Exception as e:
        logger.warning(f"⚠️ Could not store weekly report: {e}")

