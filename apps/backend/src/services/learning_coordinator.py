#!/usr/bin/env python3
"""
Learning Coordinator
Central hub that connects signal outcomes to learning system
Prioritizes 9-head weight optimization with cascading improvements
"""

import asyncio
import logging
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import asyncpg
from decimal import Decimal
import numpy as np

logger = logging.getLogger(__name__)

class LearningCoordinator:
    """
    Central learning coordinator that processes signal outcomes
    and triggers updates to head weights, thresholds, and other parameters
    
    Priority Order:
    1. 9-Head consensus weights (HIGHEST IMPACT)
    2. Indicator weights (multiplier effect)
    3. Confidence thresholds (regime-specific)
    4. Pattern effectiveness (long-term)
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        
        # Learning configuration
        self.config = {
            'learning_rate': 0.05,  # 5% adjustment per outcome
            'ema_alpha': 0.05,  # Exponential moving average alpha
            'min_outcomes_for_update': 10,  # Minimum outcomes before update
            'max_weight_change': 0.20,  # Maximum 20% change per update
            'enable_auto_update': True
        }
        
        # Statistics
        self.stats = {
            'outcomes_processed': 0,
            'head_weight_updates': 0,
            'indicator_weight_updates': 0,
            'threshold_updates': 0,
            'learning_events_logged': 0,
            'last_update_time': None
        }
        
        # Cache current weights (loaded from database)
        self.current_head_weights = {}
        self.current_thresholds = {}
        
        logger.info("🧠 Learning Coordinator initialized")
    
    async def initialize(self):
        """
        Initialize learning coordinator - load current state from database
        """
        try:
            await self._load_learning_config()
            await self._load_current_head_weights()
            await self._load_current_thresholds()
            logger.info("✅ Learning Coordinator initialized with database state")
        except Exception as e:
            logger.error(f"❌ Error initializing learning coordinator: {e}")
            # Use defaults if database load fails
            self.current_head_weights = self._get_default_head_weights()
            self.current_thresholds = {'global_threshold': 0.70}
    
    async def process_signal_outcome(self, signal_id: str, outcome_data: Dict[str, Any]):
        """
        Main entry point: Process a signal outcome and trigger learning
        
        Args:
            signal_id: The signal identifier
            outcome_data: Dictionary containing:
                - outcome_type: 'TP_HIT', 'SL_HIT', 'TIME_EXIT'
                - profit_loss_pct: Percentage profit/loss
                - sde_consensus: JSONB with 9-head votes
                - agreeing_heads: Number of heads that agreed
                - confidence, quality_score, pattern_type, etc.
        """
        try:
            logger.info(f"🔄 Processing outcome for {signal_id}: {outcome_data['outcome_type']}")
            
            # Log the outcome
            await self._log_outcome(signal_id, outcome_data)
            
            # Determine if outcome was successful
            is_win = outcome_data['profit_loss_pct'] > 0
            
            # PRIORITY 1: Update 9-head weights
            await self._update_head_weights_from_outcome(outcome_data, is_win)
            
            # PRIORITY 2: Update indicator weights (if enough data)
            if self.stats['outcomes_processed'] % 20 == 0:  # Every 20 outcomes
                await self._update_indicator_weights()
            
            # PRIORITY 3: Check if threshold adjustment needed
            if self.stats['outcomes_processed'] % 50 == 0:  # Every 50 outcomes
                await self._check_threshold_adjustment()
            
            # Update statistics
            self.stats['outcomes_processed'] += 1
            self.stats['last_update_time'] = datetime.now(timezone.utc)
            
            logger.info(f"✅ Learning completed for {signal_id}: Win={is_win}, "
                       f"P/L={outcome_data['profit_loss_pct']:.2f}%")
            
        except Exception as e:
            logger.error(f"❌ Error processing outcome for {signal_id}: {e}")
    
    async def process_rejection_outcome(self, shadow_id: str, rejection_data: Dict[str, Any]):
        """
        Process a rejection outcome for counterfactual learning
        Learn from signals that were rejected but would have won/lost
        
        Args:
            shadow_id: The shadow signal identifier
            rejection_data: Dictionary containing:
                - outcome_type: 'MISSED_OPPORTUNITY' or 'GOOD_REJECTION'
                - simulated_profit_pct: What the profit would have been
                - sde_consensus: Head votes at rejection time
                - rejection_reason: Why it was rejected
        """
        try:
            outcome_type = rejection_data['outcome_type']
            logger.info(f"🔄 Processing rejection outcome: {shadow_id} - {outcome_type}")
            
            # Determine if this was a good or bad rejection
            is_missed_opportunity = (outcome_type == 'MISSED_OPPORTUNITY')
            
            # Update head weights based on rejection learning
            await self._update_head_weights_from_rejection(rejection_data, is_missed_opportunity)
            
            # Update statistics
            self.stats['outcomes_processed'] += 1
            
            logger.info(f"✅ Rejection learning completed for {shadow_id}: "
                       f"Outcome={outcome_type}, Would have P/L={rejection_data.get('simulated_profit_pct', 0):.2f}%")
            
        except Exception as e:
            logger.error(f"❌ Error processing rejection outcome for {shadow_id}: {e}")
    
    async def _update_head_weights_from_rejection(self, rejection_data: Dict, is_missed_opportunity: bool):
        """
        Update head weights based on rejection outcome
        
        Logic for MISSED OPPORTUNITY (rejected but would have won):
        - Heads that WANTED the signal → increase weight (they were right!)
        - Heads that REJECTED the signal → decrease weight (they were wrong!)
        
        Logic for GOOD REJECTION (rejected and would have lost):
        - Heads that WANTED the signal → decrease weight (they were wrong!)
        - Heads that REJECTED the signal → increase weight (they were right!)
        
        This is INVERSE logic from normal signal outcomes
        """
        try:
            sde_consensus = rejection_data.get('sde_consensus')
            if not sde_consensus:
                logger.warning("No SDE consensus data for rejection learning")
                return
            
            if isinstance(sde_consensus, str):
                sde_consensus = json.loads(sde_consensus)
            
            heads = sde_consensus.get('heads', {})
            if not heads:
                logger.warning("No individual head votes in rejection consensus")
                return
            
            # Load current weights
            current_weights = await self._get_current_head_weights()
            new_weights = current_weights.copy()
            
            # Rejection learning rate (smaller than acceptance learning)
            learning_rate = self.config['learning_rate'] * 0.5  # Half rate for rejections
            
            # The signal direction (what was attempted)
            attempted_direction = rejection_data.get('direction', '').upper()
            
            for head_name, head_data in heads.items():
                if isinstance(head_data, dict):
                    head_direction = head_data.get('direction', 'FLAT')
                    head_confidence = head_data.get('confidence', 0.0)
                    
                    # Did this head WANT the signal?
                    head_wanted_signal = (head_direction == attempted_direction)
                    
                    # Calculate adjustment based on rejection outcome
                    if is_missed_opportunity:
                        # Signal was rejected but would have won
                        if head_wanted_signal:
                            # Head wanted it → increase weight (they saw the opportunity!)
                            adjustment = learning_rate * head_confidence
                            new_weights[head_name] = current_weights.get(head_name, 0.111) + adjustment
                        else:
                            # Head rejected it → decrease weight (they caused us to miss!)
                            adjustment = learning_rate * 0.5
                            new_weights[head_name] = current_weights.get(head_name, 0.111) - adjustment
                    
                    else:  # Good rejection
                        # Signal was rejected and would have lost
                        if head_wanted_signal:
                            # Head wanted it → decrease weight (they would have caused a loss!)
                            adjustment = learning_rate * head_confidence
                            new_weights[head_name] = current_weights.get(head_name, 0.111) - adjustment
                        else:
                            # Head rejected it → increase weight (good call!)
                            adjustment = learning_rate * 0.5
                            new_weights[head_name] = current_weights.get(head_name, 0.111) + adjustment
            
            # Apply bounds (5% - 30%)
            for head_name in new_weights:
                new_weights[head_name] = max(0.05, min(0.30, new_weights[head_name]))
            
            # Normalize
            total_weight = sum(new_weights.values())
            if total_weight > 0:
                new_weights = {k: v / total_weight for k, v in new_weights.items()}
            
            # Check if change is significant
            max_change = max(abs(new_weights[k] - current_weights.get(k, 0.111)) 
                           for k in new_weights.keys())
            
            if max_change > 0.01:
                # Store new weights
                await self._store_head_weights(new_weights, {
                    'trigger': 'rejection_learning',
                    'shadow_id': rejection_data.get('shadow_id'),
                    'outcome': 'missed_opportunity' if is_missed_opportunity else 'good_rejection',
                    'max_weight_change': float(max_change)
                })
                
                self.current_head_weights = new_weights
                self.stats['head_weight_updates'] += 1
                
                logger.info(f"✅ Head weights updated from rejection: Max change={max_change:.4f}")
                
        except Exception as e:
            logger.error(f"❌ Error updating weights from rejection: {e}")
    
    async def _update_head_weights_from_outcome(self, outcome_data: Dict, is_win: bool):
        """
        Update 9-head weights based on signal outcome
        
        Logic:
        - If head agreed AND signal won -> increase weight
        - If head agreed AND signal lost -> decrease weight
        - If head disagreed AND signal won -> decrease weight (missed opportunity)
        - If head disagreed AND signal lost -> increase weight (good rejection)
        
        Uses exponential moving average for smooth transitions
        """
        try:
            sde_consensus = outcome_data.get('sde_consensus')
            if not sde_consensus:
                logger.warning("No SDE consensus data available for learning")
                return
            
            # Parse consensus data
            if isinstance(sde_consensus, str):
                sde_consensus = json.loads(sde_consensus)
            
            # Get individual head votes
            heads = sde_consensus.get('heads', {})
            if not heads:
                logger.warning("No individual head votes found in consensus")
                return
            
            # Load current weights
            current_weights = await self._get_current_head_weights()
            new_weights = current_weights.copy()
            
            # Update each head's weight based on performance
            learning_rate = self.config['learning_rate']
            
            for head_name, head_data in heads.items():
                if isinstance(head_data, dict):
                    head_direction = head_data.get('direction', 'FLAT')
                    head_confidence = head_data.get('confidence', 0.0)
                    
                    # Determine if head agreed with signal
                    signal_direction = outcome_data.get('direction', '').upper()
                    head_agreed = (head_direction == signal_direction) or (head_direction in ['LONG', 'SHORT'])
                    
                    # Calculate weight adjustment
                    if head_agreed and is_win:
                        # Correct prediction -> increase weight
                        adjustment = learning_rate * head_confidence
                        new_weights[head_name] = current_weights.get(head_name, 0.111) + adjustment
                        
                    elif head_agreed and not is_win:
                        # Incorrect prediction -> decrease weight
                        adjustment = learning_rate * head_confidence
                        new_weights[head_name] = current_weights.get(head_name, 0.111) - adjustment
                        
                    elif not head_agreed and is_win:
                        # Missed opportunity -> small decrease
                        adjustment = learning_rate * 0.5  # Half adjustment for disagreement
                        new_weights[head_name] = current_weights.get(head_name, 0.111) - adjustment
                        
                    elif not head_agreed and not is_win:
                        # Good rejection -> small increase
                        adjustment = learning_rate * 0.5
                        new_weights[head_name] = current_weights.get(head_name, 0.111) + adjustment
            
            # Apply bounds (no weight > 0.30 or < 0.05)
            for head_name in new_weights:
                new_weights[head_name] = max(0.05, min(0.30, new_weights[head_name]))
            
            # Normalize weights to sum to 1.0
            total_weight = sum(new_weights.values())
            if total_weight > 0:
                new_weights = {k: v / total_weight for k, v in new_weights.items()}
            
            # Check if change is significant (> 1%)
            max_change = max(abs(new_weights[k] - current_weights.get(k, 0.111)) 
                           for k in new_weights.keys())
            
            if max_change > 0.01:  # Only update if change > 1%
                # Store new weights
                await self._store_head_weights(new_weights, {
                    'trigger': 'signal_outcome',
                    'signal_id': outcome_data['signal_id'],
                    'outcome': 'win' if is_win else 'loss',
                    'max_weight_change': float(max_change)
                })
                
                self.current_head_weights = new_weights
                self.stats['head_weight_updates'] += 1
                
                logger.info(f"✅ Head weights updated: Max change={max_change:.4f}")
                
                # Log significant weight changes
                for head_name, new_weight in new_weights.items():
                    old_weight = current_weights.get(head_name, 0.111)
                    change = new_weight - old_weight
                    if abs(change) > 0.02:  # Log changes > 2%
                        logger.info(f"   {head_name}: {old_weight:.4f} → {new_weight:.4f} "
                                  f"({'+' if change > 0 else ''}{change:.4f})")
            
        except Exception as e:
            logger.error(f"❌ Error updating head weights: {e}")
    
    async def _update_indicator_weights(self):
        """
        Update indicator weights based on recent performance
        (Simplified implementation - full version would analyze indicator contribution)
        """
        try:
            logger.info("Indicator weight update would run here (Phase 1B)")
            self.stats['indicator_weight_updates'] += 1
        except Exception as e:
            logger.error(f"❌ Error updating indicator weights: {e}")
    
    async def _check_threshold_adjustment(self):
        """
        Check if confidence thresholds need adjustment
        (Simplified implementation - full version would analyze win rates per regime)
        """
        try:
            logger.info("Threshold adjustment check would run here (Phase 1B)")
            self.stats['threshold_updates'] += 1
        except Exception as e:
            logger.error(f"❌ Error checking threshold adjustment: {e}")
    
    async def _load_learning_config(self):
        """
        Load learning configuration from database
        """
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT state_data
                    FROM active_learning_state
                    WHERE state_type = 'learning_config'
                """)
                
                if result:
                    config_data = result['state_data']
                    if isinstance(config_data, str):
                        config_data = json.loads(config_data)
                    self.config.update(config_data)
                    logger.info(f"✅ Loaded learning config: learning_rate={self.config['learning_rate']}")
                    
        except Exception as e:
            logger.warning(f"⚠️ Could not load learning config, using defaults: {e}")
    
    async def _load_current_head_weights(self):
        """
        Load current head weights from database
        """
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT state_data
                    FROM active_learning_state
                    WHERE state_type = 'head_weights'
                """)
                
                if result:
                    weights_data = result['state_data']
                    if isinstance(weights_data, str):
                        weights_data = json.loads(weights_data)
                    self.current_head_weights = weights_data
                    logger.info(f"✅ Loaded head weights from database")
                else:
                    self.current_head_weights = self._get_default_head_weights()
                    
        except Exception as e:
            logger.warning(f"⚠️ Could not load head weights, using defaults: {e}")
            self.current_head_weights = self._get_default_head_weights()
    
    async def _load_current_thresholds(self):
        """
        Load current thresholds from database
        """
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT state_data
                    FROM active_learning_state
                    WHERE state_type = 'confidence_threshold'
                """)
                
                if result:
                    threshold_data = result['state_data']
                    if isinstance(threshold_data, str):
                        threshold_data = json.loads(threshold_data)
                    self.current_thresholds = threshold_data
                    logger.info(f"✅ Loaded thresholds from database")
                    
        except Exception as e:
            logger.warning(f"⚠️ Could not load thresholds, using defaults: {e}")
            self.current_thresholds = {'global_threshold': 0.70}
    
    async def _get_current_head_weights(self) -> Dict[str, float]:
        """
        Get current head weights (from cache or database)
        """
        if not self.current_head_weights:
            await self._load_current_head_weights()
        return self.current_head_weights
    
    async def _store_head_weights(self, new_weights: Dict[str, float], metrics: Dict):
        """
        Store new head weights in database with versioning
        """
        try:
            async with self.db_pool.acquire() as conn:
                # Use the database function for atomic update
                await conn.execute("""
                    SELECT update_head_weights($1::jsonb, $2::jsonb)
                """, json.dumps(new_weights), json.dumps(metrics))
                
                logger.info("✅ Head weights stored in database")
                
        except Exception as e:
            logger.error(f"❌ Error storing head weights: {e}")
    
    async def _log_outcome(self, signal_id: str, outcome_data: Dict):
        """
        Log outcome processing for audit trail
        """
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO learning_events (
                        event_type, signal_id, state_type, 
                        new_value, triggered_by, notes
                    )
                    VALUES ($1, $2, $3, $4, $5, $6)
                """,
                'outcome_processed',
                signal_id,
                'signal_outcome',
                json.dumps({
                    'outcome_type': outcome_data['outcome_type'],
                    'profit_loss_pct': outcome_data['profit_loss_pct'],
                    'confidence': outcome_data.get('confidence')
                }),
                'outcome_monitor',
                f"Processed {outcome_data['outcome_type']} for learning"
                )
                
                self.stats['learning_events_logged'] += 1
                
        except Exception as e:
            logger.warning(f"⚠️ Could not log learning event: {e}")
    
    def _get_default_head_weights(self) -> Dict[str, float]:
        """
        Get default head weights (equal distribution)
        """
        return {
            'HEAD_A': 0.111,
            'HEAD_B': 0.111,
            'HEAD_C': 0.111,
            'HEAD_D': 0.111,
            'HEAD_E': 0.111,
            'HEAD_F': 0.111,
            'HEAD_G': 0.111,
            'HEAD_H': 0.111,
            'HEAD_I': 0.111
        }
    
    async def get_current_head_weights(self) -> Dict[str, float]:
        """
        Public method to get current head weights
        """
        if not self.current_head_weights:
            await self._load_current_head_weights()
        return self.current_head_weights
    
    def get_stats(self) -> Dict:
        """
        Get learning coordinator statistics
        """
        return {
            **self.stats,
            'current_head_weights': self.current_head_weights,
            'learning_config': self.config
        }

