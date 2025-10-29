#!/usr/bin/env python3
"""
Adaptive Learning Engine for Advanced Pattern Recognition
Tracks pattern performance and adapts confidence based on market feedback
"""

import logging
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json

logger = logging.getLogger(__name__)

class AdaptiveLearningEngine:
    """Engine for adaptive learning based on pattern performance"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        self.initialized = False
        
        # Learning parameters
        self.learning_config = {
            'min_samples': 10,  # Minimum samples for learning
            'learning_rate': 0.1,  # How fast to adapt
            'confidence_decay': 0.95,  # Confidence decay factor
            'performance_window': 30,  # Days to look back for performance
            'regime_adaptation': True,  # Adapt based on market regime
            'pattern_specific': True  # Pattern-specific learning
        }
        
        logger.info("🔧 Adaptive Learning Engine initialized")
    
    async def initialize(self):
        """Initialize the adaptive learning engine"""
        try:
            logger.info("🔧 Initializing Adaptive Learning Engine...")
            
            # Connect to database
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            
            self.initialized = True
            logger.info("✅ Adaptive Learning Engine ready")
            
        except Exception as e:
            logger.error(f"❌ Adaptive Learning Engine initialization failed: {e}")
            raise
    
    async def track_pattern_outcome(self, pattern_data: Dict[str, Any], outcome: str, 
                                  outcome_price: float, profit_loss: float = None) -> bool:
        """
        Track the outcome of a pattern prediction
        
        Args:
            pattern_data: Original pattern data
            outcome: 'success', 'failure', or 'neutral'
            outcome_price: Price at outcome determination
            profit_loss: P&L if applicable
            
        Returns:
            Success status
        """
        try:
            if not self.initialized:
                await self.initialize()
            
            # Get tracking ID from pattern data
            tracking_id = pattern_data.get('tracking_id')
            if not tracking_id:
                logger.warning("No tracking_id found in pattern data")
                return False
            
            # Update pattern performance tracking
            await self._update_pattern_outcome(tracking_id, outcome, outcome_price, profit_loss)
            
            # Trigger adaptive learning
            await self._trigger_adaptive_learning(pattern_data)
            
            logger.info(f"✅ Tracked pattern outcome: {outcome} for {tracking_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to track pattern outcome: {e}")
            return False
    
    async def _update_pattern_outcome(self, tracking_id: str, outcome: str, 
                                    outcome_price: float, profit_loss: float = None):
        """Update pattern outcome in database"""
        try:
            self.cursor.execute("""
                UPDATE pattern_performance_tracking
                SET actual_outcome = %s,
                    outcome_timestamp = NOW(),
                    outcome_price = %s,
                    profit_loss = %s
                WHERE tracking_id = %s
            """, (outcome, outcome_price, profit_loss, tracking_id))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"❌ Failed to update pattern outcome: {e}")
            self.conn.rollback()
    
    async def _trigger_adaptive_learning(self, pattern_data: Dict[str, Any]):
        """Trigger adaptive learning for the pattern"""
        try:
            # Get recent performance for this pattern type and regime
            pattern_name = pattern_data.get('pattern_name')
            market_regime = pattern_data.get('market_regime', 'unknown')
            
            # Calculate performance metrics
            performance_metrics = await self._calculate_performance_metrics(pattern_name, market_regime)
            
            # Update adaptive confidence model
            await self._update_confidence_model(pattern_name, market_regime, performance_metrics)
            
        except Exception as e:
            logger.error(f"❌ Adaptive learning failed: {e}")
    
    async def _calculate_performance_metrics(self, pattern_name: str, market_regime: str) -> Dict[str, Any]:
        """Calculate performance metrics for pattern and regime"""
        try:
            # Get recent performance data
            self.cursor.execute("""
                SELECT actual_outcome, pattern_confidence, profit_loss, created_at
                FROM pattern_performance_tracking
                WHERE pattern_name = %s 
                AND market_regime = %s
                AND actual_outcome IS NOT NULL
                AND created_at > NOW() - INTERVAL '%s days'
                ORDER BY created_at DESC
            """, (pattern_name, market_regime, self.learning_config['performance_window']))
            
            results = self.cursor.fetchall()
            
            if len(results) < self.learning_config['min_samples']:
                return {
                    'accuracy': 0.5,
                    'precision': 0.5,
                    'recall': 0.5,
                    'f1_score': 0.5,
                    'avg_profit_loss': 0.0,
                    'success_rate': 0.5,
                    'sample_count': len(results)
                }
            
            # Calculate metrics
            outcomes = [row[0] for row in results]
            confidences = [row[1] for row in results]
            profits = [row[2] for row in results if row[2] is not None]
            
            # Success rate
            success_count = outcomes.count('success')
            success_rate = success_count / len(outcomes)
            
            # Accuracy (overall correctness)
            accuracy = success_rate
            
            # Precision (success rate among positive predictions)
            positive_predictions = [i for i, conf in enumerate(confidences) if conf > 0.6]
            if positive_predictions:
                positive_successes = sum(1 for i in positive_predictions if outcomes[i] == 'success')
                precision = positive_successes / len(positive_predictions)
            else:
                precision = 0.5
            
            # Recall (success rate among actual successes)
            if success_count > 0:
                high_confidence_successes = sum(1 for i, outcome in enumerate(outcomes) 
                                              if outcome == 'success' and confidences[i] > 0.6)
                recall = high_confidence_successes / success_count
            else:
                recall = 0.5
            
            # F1 Score
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Average profit/loss
            avg_profit_loss = sum(profits) / len(profits) if profits else 0.0
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'avg_profit_loss': avg_profit_loss,
                'success_rate': success_rate,
                'sample_count': len(results)
            }
            
        except Exception as e:
            logger.error(f"❌ Performance metrics calculation failed: {e}")
            return {
                'accuracy': 0.5,
                'precision': 0.5,
                'recall': 0.5,
                'f1_score': 0.5,
                'avg_profit_loss': 0.0,
                'success_rate': 0.5,
                'sample_count': 0
            }
    
    async def _update_confidence_model(self, pattern_name: str, market_regime: str, 
                                     performance_metrics: Dict[str, Any]):
        """Update confidence model based on performance"""
        try:
            # Get current model
            current_model = await self._get_current_model(pattern_name, market_regime)
            
            # Calculate new feature weights
            new_weights = await self._calculate_new_weights(current_model, performance_metrics)
            
            # Create new model version
            model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Store new model
            await self._store_confidence_model(pattern_name, market_regime, model_version, 
                                             new_weights, performance_metrics)
            
            # Deactivate old model
            if current_model:
                await self._deactivate_model(current_model['model_id'])
            
            logger.info(f"✅ Updated confidence model for {pattern_name} in {market_regime} regime")
            
        except Exception as e:
            logger.error(f"❌ Confidence model update failed: {e}")
    
    async def _get_current_model(self, pattern_name: str, market_regime: str) -> Optional[Dict[str, Any]]:
        """Get current active model for pattern and regime"""
        try:
            self.cursor.execute("""
                SELECT model_id, feature_weights, performance_metrics, validation_score
                FROM adaptive_confidence_models
                WHERE pattern_name = %s 
                AND market_regime = %s 
                AND is_active = TRUE
                ORDER BY created_at DESC
                LIMIT 1
            """, (pattern_name, market_regime))
            
            result = self.cursor.fetchone()
            
            if result:
                return {
                    'model_id': result[0],
                    'feature_weights': result[1],
                    'performance_metrics': result[2],
                    'validation_score': result[3]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get current model: {e}")
            return None
    
    async def _calculate_new_weights(self, current_model: Optional[Dict[str, Any]], 
                                   performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate new feature weights based on performance"""
        try:
            # Default weights
            default_weights = {
                'technical_score': 0.4,
                'volume_confirmation': 0.2,
                'trend_alignment': 0.2,
                'market_regime': 0.1,
                'historical_success': 0.1
            }
            
            if not current_model:
                return default_weights
            
            current_weights = current_model['feature_weights']
            learning_rate = self.learning_config['learning_rate']
            
            # Adjust weights based on performance
            performance_score = performance_metrics['f1_score']
            
            # If performance is good, increase historical success weight
            if performance_score > 0.7:
                adjustment = learning_rate * (performance_score - 0.5)
                current_weights['historical_success'] = min(0.3, current_weights['historical_success'] + adjustment)
                current_weights['technical_score'] = max(0.3, current_weights['technical_score'] - adjustment * 0.5)
            
            # If performance is poor, increase technical score weight
            elif performance_score < 0.4:
                adjustment = learning_rate * (0.5 - performance_score)
                current_weights['technical_score'] = min(0.6, current_weights['technical_score'] + adjustment)
                current_weights['historical_success'] = max(0.05, current_weights['historical_success'] - adjustment * 0.5)
            
            # Normalize weights
            total_weight = sum(current_weights.values())
            normalized_weights = {k: v / total_weight for k, v in current_weights.items()}
            
            return normalized_weights
            
        except Exception as e:
            logger.error(f"❌ Weight calculation failed: {e}")
            return default_weights
    
    async def _store_confidence_model(self, pattern_name: str, market_regime: str, 
                                    model_version: str, feature_weights: Dict[str, Any], 
                                    performance_metrics: Dict[str, Any]):
        """Store new confidence model in database"""
        try:
            model_id = f"model_{pattern_name}_{market_regime}_{model_version}"
            
            self.cursor.execute("""
                INSERT INTO adaptive_confidence_models (
                    timestamp, model_id, pattern_name, market_regime, model_version,
                    model_type, feature_weights, performance_metrics, training_data_size,
                    last_training_timestamp, validation_score, is_active
                ) VALUES (
                    NOW(), %s, %s, %s, %s,
                    'ensemble', %s::jsonb, %s::jsonb, %s,
                    NOW(), %s, TRUE
                )
            """, (
                model_id,
                pattern_name,
                market_regime,
                model_version,
                json.dumps(feature_weights),
                json.dumps(performance_metrics),
                performance_metrics['sample_count'],
                performance_metrics['f1_score']
            ))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"❌ Failed to store confidence model: {e}")
            self.conn.rollback()
    
    async def _deactivate_model(self, model_id: str):
        """Deactivate old model"""
        try:
            self.cursor.execute("""
                UPDATE adaptive_confidence_models
                SET is_active = FALSE
                WHERE model_id = %s
            """, (model_id,))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"❌ Failed to deactivate model: {e}")
            self.conn.rollback()
    
    async def get_adaptive_confidence(self, pattern_data: Dict[str, Any], 
                                    base_confidence: float) -> float:
        """
        Get adaptive confidence score for a pattern
        
        Args:
            pattern_data: Pattern data
            base_confidence: Base confidence score
            
        Returns:
            Adaptive confidence score
        """
        try:
            if not self.initialized:
                await self.initialize()
            
            pattern_name = pattern_data.get('pattern_name')
            market_regime = pattern_data.get('market_regime', 'unknown')
            
            # Get current model
            current_model = await self._get_current_model(pattern_name, market_regime)
            
            if not current_model:
                return base_confidence
            
            # Get recent performance
            performance_metrics = await self._calculate_performance_metrics(pattern_name, market_regime)
            
            # Calculate adaptive confidence
            adaptive_confidence = await self._calculate_adaptive_confidence(
                base_confidence, current_model, performance_metrics
            )
            
            return adaptive_confidence
            
        except Exception as e:
            logger.error(f"❌ Adaptive confidence calculation failed: {e}")
            return base_confidence
    
    async def _calculate_adaptive_confidence(self, base_confidence: float, 
                                           current_model: Dict[str, Any], 
                                           performance_metrics: Dict[str, Any]) -> float:
        """Calculate adaptive confidence score"""
        try:
            feature_weights = current_model['feature_weights']
            performance_score = performance_metrics['f1_score']
            
            # Calculate weighted confidence
            technical_weight = feature_weights.get('technical_score', 0.4)
            historical_weight = feature_weights.get('historical_success', 0.1)
            
            # Technical confidence (base confidence)
            technical_confidence = base_confidence * technical_weight
            
            # Historical confidence (based on recent performance)
            historical_confidence = performance_score * historical_weight
            
            # Other factors (market regime, volume, etc.)
            other_confidence = base_confidence * (1 - technical_weight - historical_weight)
            
            # Combine confidences
            adaptive_confidence = technical_confidence + historical_confidence + other_confidence
            
            # Apply confidence decay for poor performance
            if performance_score < 0.4:
                decay_factor = self.learning_config['confidence_decay']
                adaptive_confidence *= decay_factor
            
            # Ensure confidence is within bounds
            adaptive_confidence = max(0.0, min(1.0, adaptive_confidence))
            
            return adaptive_confidence
            
        except Exception as e:
            logger.error(f"❌ Adaptive confidence calculation failed: {e}")
            return base_confidence
    
    async def get_performance_summary(self, pattern_name: str = None, 
                                    market_regime: str = None) -> Dict[str, Any]:
        """Get performance summary for patterns"""
        try:
            if not self.initialized:
                await self.initialize()
            
            # Build query
            query = """
                SELECT pattern_name, market_regime, 
                       COUNT(*) as total_patterns,
                       COUNT(CASE WHEN actual_outcome = 'success' THEN 1 END) as successes,
                       COUNT(CASE WHEN actual_outcome = 'failure' THEN 1 END) as failures,
                       AVG(pattern_confidence) as avg_confidence,
                       AVG(profit_loss) as avg_profit_loss
                FROM pattern_performance_tracking
                WHERE actual_outcome IS NOT NULL
            """
            
            params = []
            if pattern_name:
                query += " AND pattern_name = %s"
                params.append(pattern_name)
            
            if market_regime:
                query += " AND market_regime = %s"
                params.append(market_regime)
            
            query += " GROUP BY pattern_name, market_regime ORDER BY total_patterns DESC"
            
            self.cursor.execute(query, params)
            results = self.cursor.fetchall()
            
            summary = []
            for row in results:
                pattern_name, regime, total, successes, failures, avg_conf, avg_pl = row
                success_rate = successes / total if total > 0 else 0
                
                summary.append({
                    'pattern_name': pattern_name,
                    'market_regime': regime,
                    'total_patterns': total,
                    'successes': successes,
                    'failures': failures,
                    'success_rate': success_rate,
                    'avg_confidence': avg_conf,
                    'avg_profit_loss': avg_pl
                })
            
            return {'summary': summary}
            
        except Exception as e:
            logger.error(f"❌ Performance summary failed: {e}")
            return {'summary': []}
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
            logger.info("✅ Adaptive Learning Engine cleaned up")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

# Example usage
async def test_adaptive_learning_engine():
    """Test the adaptive learning engine"""
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'alphapulse',
        'user': 'postgres',
        'password': 'Emon_@17711'
    }
    
    # Create adaptive learning engine
    learning_engine = AdaptiveLearningEngine(db_config)
    
    try:
        # Initialize engine
        await learning_engine.initialize()
        
        # Sample pattern data
        pattern_data = {
            'tracking_id': 'test_tracking_123',
            'pattern_name': 'doji',
            'market_regime': 'trending',
            'confidence': 0.75
        }
        
        # Test tracking outcome
        success = await learning_engine.track_pattern_outcome(
            pattern_data, 'success', 51000.0, 500.0
        )
        
        print(f"🎯 Pattern Outcome Tracking:")
        print(f"   Success: {success}")
        
        # Test adaptive confidence
        adaptive_confidence = await learning_engine.get_adaptive_confidence(
            pattern_data, 0.75
        )
        
        print(f"   Base Confidence: 0.75")
        print(f"   Adaptive Confidence: {adaptive_confidence:.3f}")
        
        # Test performance summary
        performance_summary = await learning_engine.get_performance_summary()
        
        print(f"📊 Performance Summary:")
        for item in performance_summary['summary']:
            print(f"   {item['pattern_name']} ({item['market_regime']}): "
                  f"{item['success_rate']:.2f} success rate, "
                  f"{item['total_patterns']} patterns")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
    
    finally:
        await learning_engine.cleanup()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_adaptive_learning_engine())
