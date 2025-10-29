#!/usr/bin/env python3
"""
Daily Learning Job
Runs incremental learning updates every day at midnight UTC
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
import asyncpg
import json

logger = logging.getLogger(__name__)

async def run_daily_learning(db_pool: asyncpg.Pool):
    """
    Daily learning job - small incremental updates
    Runs at 00:00 UTC every day
    """
    logger.info("="*80)
    logger.info("🌙 DAILY LEARNING JOB STARTED")
    logger.info(f"Time: {datetime.now(timezone.utc).isoformat()}")
    logger.info("="*80)
    
    try:
        # 1. Get yesterday's outcomes (last 24 hours)
        outcomes = await _get_outcomes_last_24h(db_pool)
        logger.info(f"📊 Retrieved {len(outcomes)} outcomes from last 24 hours")
        
        if len(outcomes) == 0:
            logger.info("ℹ️ No outcomes to process - skipping daily learning")
            return {
                'status': 'skipped',
                'reason': 'no_outcomes',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        # 2. Calculate daily performance metrics
        daily_metrics = await _calculate_daily_metrics(outcomes)
        logger.info(f"📈 Daily Performance:")
        logger.info(f"   Win Rate: {daily_metrics['win_rate']:.2%}")
        logger.info(f"   Avg Profit: {daily_metrics['avg_profit']:.2f}%")
        logger.info(f"   Total Signals: {daily_metrics['total_signals']}")
        
        # 3. Update head weights if enough data
        weight_updates = None
        if len(outcomes) >= 10:
            logger.info("🧠 Updating head weights (sufficient data)...")
            weight_updates = await _update_head_weights_incremental(db_pool, outcomes)
            logger.info(f"✅ Head weights updated: {weight_updates['changes_made']} heads adjusted")
        else:
            logger.info(f"⏭️ Skipping head weight update (need 10+ outcomes, have {len(outcomes)})")
        
        # 4. Store daily report
        report = {
            'date': datetime.now(timezone.utc).date().isoformat(),
            'outcomes_processed': len(outcomes),
            'daily_metrics': daily_metrics,
            'weight_updates': weight_updates,
            'job_completed_at': datetime.now(timezone.utc).isoformat()
        }
        
        await _store_daily_report(db_pool, report)
        logger.info("💾 Daily report stored in database")
        
        logger.info("="*80)
        logger.info("✅ DAILY LEARNING JOB COMPLETED")
        logger.info("="*80)
        
        return report
        
    except Exception as e:
        logger.error(f"❌ Daily learning job failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

async def _get_outcomes_last_24h(db_pool: asyncpg.Pool) -> List[Dict]:
    """Get all signal outcomes from last 24 hours"""
    try:
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    signal_id, symbol, direction, outcome,
                    profit_loss_pct, confidence, quality_score,
                    sde_consensus, pattern_type, indicators,
                    signal_timestamp, completed_at
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

async def _calculate_daily_metrics(outcomes: List[Dict]) -> Dict[str, Any]:
    """Calculate daily performance metrics"""
    try:
        if not outcomes:
            return {
                'total_signals': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0
            }
        
        total = len(outcomes)
        wins = len([o for o in outcomes if o['outcome'] == 'win'])
        losses = len([o for o in outcomes if o['outcome'] == 'loss'])
        
        win_rate = wins / total if total > 0 else 0.0
        
        profits = [float(o['profit_loss_pct']) for o in outcomes if o['profit_loss_pct'] is not None]
        avg_profit = sum(profits) / len(profits) if profits else 0.0
        
        return {
            'total_signals': total,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'total_profit': sum(profits) if profits else 0.0
        }
        
    except Exception as e:
        logger.error(f"❌ Error calculating metrics: {e}")
        return {}

async def _update_head_weights_incremental(
    db_pool: asyncpg.Pool, 
    outcomes: List[Dict]
) -> Dict[str, Any]:
    """
    Update head weights incrementally based on recent outcomes
    Uses exponential moving average for smooth updates
    """
    try:
        # Get current weights
        async with db_pool.acquire() as conn:
            result = await conn.fetchrow("""
                SELECT state_data
                FROM active_learning_state
                WHERE state_type = 'head_weights'
            """)
            
            if not result:
                logger.warning("⚠️ No current head weights found")
                return {'changes_made': 0}
            
            current_weights = result['state_data']
            if isinstance(current_weights, str):
                current_weights = json.loads(current_weights)
        
        # Calculate performance by head
        head_performance = {}
        head_names = ['HEAD_A', 'HEAD_B', 'HEAD_C', 'HEAD_D', 'HEAD_E', 
                     'HEAD_F', 'HEAD_G', 'HEAD_H', 'HEAD_I']
        
        for head_name in head_names:
            head_performance[head_name] = {
                'agreements': 0,
                'wins_when_agreed': 0,
                'losses_when_agreed': 0
            }
        
        # Analyze each outcome
        for outcome in outcomes:
            is_win = (outcome['outcome'] == 'win')
            sde_consensus = outcome.get('sde_consensus')
            
            if not sde_consensus:
                continue
                
            if isinstance(sde_consensus, str):
                sde_consensus = json.loads(sde_consensus)
            
            heads = sde_consensus.get('heads', {})
            signal_direction = outcome['direction'].upper()
            
            for head_name in head_names:
                if head_name in heads:
                    head_data = heads[head_name]
                    if isinstance(head_data, dict):
                        head_direction = head_data.get('direction', 'FLAT')
                        
                        # Check if head agreed
                        if head_direction == signal_direction:
                            head_performance[head_name]['agreements'] += 1
                            if is_win:
                                head_performance[head_name]['wins_when_agreed'] += 1
                            else:
                                head_performance[head_name]['losses_when_agreed'] += 1
        
        # Calculate new weights
        new_weights = current_weights.copy()
        learning_rate = 0.03  # Smaller learning rate for daily updates (3%)
        changes_made = 0
        
        for head_name in head_names:
            perf = head_performance[head_name]
            agreements = perf['agreements']
            
            if agreements > 0:
                win_rate = perf['wins_when_agreed'] / agreements
                
                # Adjust weight based on win rate
                old_weight = current_weights.get(head_name, 0.111)
                
                if win_rate > 0.65:  # Good performance
                    adjustment = learning_rate * (win_rate - 0.5)
                    new_weights[head_name] = old_weight + adjustment
                    changes_made += 1
                elif win_rate < 0.55:  # Poor performance
                    adjustment = learning_rate * (0.5 - win_rate)
                    new_weights[head_name] = old_weight - adjustment
                    changes_made += 1
        
        # Apply bounds
        for head_name in new_weights:
            new_weights[head_name] = max(0.05, min(0.30, new_weights[head_name]))
        
        # Normalize
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            new_weights = {k: v / total_weight for k, v in new_weights.items()}
        
        # Update database if changes significant
        if changes_made > 0:
            async with db_pool.acquire() as conn:
                await conn.execute("""
                    SELECT update_head_weights($1::jsonb, $2::jsonb)
                """, 
                json.dumps(new_weights),
                json.dumps({
                    'trigger': 'daily_job',
                    'outcomes_analyzed': len(outcomes),
                    'changes_made': changes_made
                }))
            
            logger.info(f"✅ Updated {changes_made} head weights")
        
        return {
            'changes_made': changes_made,
            'new_weights': new_weights,
            'head_performance': head_performance
        }
        
    except Exception as e:
        logger.error(f"❌ Error updating head weights: {e}")
        return {'changes_made': 0, 'error': str(e)}

async def _store_daily_report(db_pool: asyncpg.Pool, report: Dict):
    """Store daily learning report in database"""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO learning_events (
                    event_type, event_timestamp, new_value, 
                    triggered_by, notes
                )
                VALUES ($1, $2, $3, $4, $5)
            """,
            'daily_learning_report',
            datetime.now(timezone.utc),
            json.dumps(report),
            'daily_job',
            f"Daily learning job completed: {report.get('outcomes_processed', 0)} outcomes processed"
            )
    except Exception as e:
        logger.warning(f"⚠️ Could not store daily report: {e}")

