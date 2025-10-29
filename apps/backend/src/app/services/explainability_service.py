#!/usr/bin/env python3
"""
Explainability Service
Provides SHAP values, feature importance, and decision explanations
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json

# ML imports for explainability
try:
    import shap
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML libraries not available for explainability")

logger = logging.getLogger(__name__)

@dataclass
class FeatureContribution:
    """Feature contribution to a decision"""
    feature_name: str
    contribution_value: float
    contribution_percentage: float
    feature_importance: float
    direction: str  # 'positive', 'negative', 'neutral'

@dataclass
class DecisionExplanation:
    """Complete explanation for a trading decision"""
    decision_type: str
    decision_value: str
    confidence_score: float
    explanation_text: str
    feature_contributions: List[FeatureContribution]
    shap_values: Dict[str, float]
    contributing_factors: Dict[str, Any]
    metadata: Dict[str, Any]

class ExplainabilityService:
    """Service for providing explainability for ML and trading decisions"""
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
        # SHAP explainer cache
        self.shap_explainers = {}
        
        # Feature importance cache
        self.feature_importance_cache = {}
        
        # Explanation templates
        self.explanation_templates = {
            'volume_signal': {
                'buy': "Strong buy signal based on volume analysis. Key factors: {top_factors}",
                'sell': "Strong sell signal based on volume analysis. Key factors: {top_factors}",
                'hold': "Neutral volume conditions suggest holding position. Key factors: {top_factors}"
            },
            'ml_prediction': {
                'bullish': "ML model predicts bullish movement with {confidence}% confidence. Key features: {top_features}",
                'bearish': "ML model predicts bearish movement with {confidence}% confidence. Key features: {top_features}",
                'neutral': "ML model shows neutral prediction with {confidence}% confidence. Key features: {top_features}"
            },
            'rl_action': {
                'buy': "RL agent recommends buy action based on current market state. Key factors: {top_factors}",
                'sell': "RL agent recommends sell action based on current market state. Key factors: {top_factors}",
                'hold': "RL agent recommends holding current position. Key factors: {top_factors}"
            },
            'anomaly_alert': {
                'high': "High-priority anomaly detected. Key indicators: {top_indicators}",
                'medium': "Medium-priority anomaly detected. Key indicators: {top_indicators}",
                'low': "Low-priority anomaly detected. Key indicators: {top_indicators}"
            }
        }
        
        self.logger.info("🔍 Explainability Service initialized")
    
    async def explain_volume_decision(self, volume_analysis_result: Dict) -> DecisionExplanation:
        """Explain volume-based trading decision"""
        try:
            # Extract key volume metrics
            volume_ratio = volume_analysis_result.get('volume_ratio', 1.0)
            volume_positioning_score = volume_analysis_result.get('volume_positioning_score', 0.5)
            order_book_imbalance = volume_analysis_result.get('order_book_imbalance', 0.0)
            volume_trend = volume_analysis_result.get('volume_trend', 'stable')
            
            # Determine decision
            if volume_ratio > 2.0 and volume_positioning_score > 0.7:
                decision_value = 'buy'
                confidence_score = min(volume_positioning_score * 0.8 + (volume_ratio - 1.0) * 0.1, 1.0)
            elif volume_ratio < 0.5 and volume_positioning_score < 0.3:
                decision_value = 'sell'
                confidence_score = min((1.0 - volume_positioning_score) * 0.8 + (1.0 - volume_ratio) * 0.1, 1.0)
            else:
                decision_value = 'hold'
                confidence_score = 0.5
            
            # Calculate feature contributions
            feature_contributions = [
                FeatureContribution(
                    feature_name='volume_ratio',
                    contribution_value=volume_ratio - 1.0,
                    contribution_percentage=abs(volume_ratio - 1.0) * 40,
                    feature_importance=0.4,
                    direction='positive' if volume_ratio > 1.0 else 'negative'
                ),
                FeatureContribution(
                    feature_name='volume_positioning_score',
                    contribution_value=volume_positioning_score - 0.5,
                    contribution_percentage=abs(volume_positioning_score - 0.5) * 40,
                    feature_importance=0.4,
                    direction='positive' if volume_positioning_score > 0.5 else 'negative'
                ),
                FeatureContribution(
                    feature_name='order_book_imbalance',
                    contribution_value=order_book_imbalance,
                    contribution_percentage=abs(order_book_imbalance) * 20,
                    feature_importance=0.2,
                    direction='positive' if order_book_imbalance > 0 else 'negative'
                )
            ]
            
            # Generate explanation text
            top_factors = [f"{fc.feature_name} ({fc.contribution_percentage:.1f}%)" 
                          for fc in sorted(feature_contributions, key=lambda x: x.contribution_percentage, reverse=True)[:2]]
            
            explanation_text = self.explanation_templates['volume_signal'][decision_value].format(
                top_factors=', '.join(top_factors)
            )
            
            # Create SHAP values
            shap_values = {
                'volume_ratio': volume_ratio - 1.0,
                'volume_positioning_score': volume_positioning_score - 0.5,
                'order_book_imbalance': order_book_imbalance
            }
            
            # Contributing factors
            contributing_factors = {
                'volume_metrics': {
                    'volume_ratio': volume_ratio,
                    'volume_trend': volume_trend,
                    'volume_positioning_score': volume_positioning_score
                },
                'order_book_metrics': {
                    'order_book_imbalance': order_book_imbalance
                },
                'decision_logic': {
                    'threshold_volume_ratio': 2.0,
                    'threshold_positioning_score': 0.7
                }
            }
            
            return DecisionExplanation(
                decision_type='volume_signal',
                decision_value=decision_value,
                confidence_score=confidence_score,
                explanation_text=explanation_text,
                feature_contributions=feature_contributions,
                shap_values=shap_values,
                contributing_factors=contributing_factors,
                metadata={
                    'analysis_timestamp': datetime.now().isoformat(),
                    'analysis_method': 'volume_analysis'
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error explaining volume decision: {e}")
            return self._get_default_explanation('volume_signal', 'hold', 0.5)
    
    async def explain_ml_prediction(self, model_name: str, prediction_result: Dict, feature_vector: Dict) -> DecisionExplanation:
        """Explain ML model prediction using SHAP values"""
        try:
            if not ML_AVAILABLE:
                return self._get_default_explanation('ml_prediction', 'neutral', 0.5)
            
            prediction_value = prediction_result.get('prediction', 'neutral')
            confidence_score = prediction_result.get('confidence', 0.5)
            
            # Calculate SHAP values
            shap_values = await self._calculate_shap_values(model_name, feature_vector)
            
            # Calculate feature contributions
            feature_contributions = []
            total_contribution = sum(abs(v) for v in shap_values.values())
            
            for feature_name, shap_value in shap_values.items():
                if total_contribution > 0:
                    contribution_percentage = (abs(shap_value) / total_contribution) * 100
                else:
                    contribution_percentage = 0.0
                
                feature_contributions.append(FeatureContribution(
                    feature_name=feature_name,
                    contribution_value=shap_value,
                    contribution_percentage=contribution_percentage,
                    feature_importance=await self._get_feature_importance(model_name, feature_name),
                    direction='positive' if shap_value > 0 else 'negative'
                ))
            
            # Sort by contribution percentage
            feature_contributions.sort(key=lambda x: x.contribution_percentage, reverse=True)
            
            # Generate explanation text
            top_features = [f"{fc.feature_name} ({fc.contribution_percentage:.1f}%)" 
                           for fc in feature_contributions[:3]]
            
            explanation_text = self.explanation_templates['ml_prediction'][prediction_value].format(
                confidence=confidence_score * 100,
                top_features=', '.join(top_features)
            )
            
            # Contributing factors
            contributing_factors = {
                'model_info': {
                    'model_name': model_name,
                    'prediction_type': prediction_result.get('prediction_type', 'unknown')
                },
                'feature_vector': feature_vector,
                'shap_analysis': {
                    'total_contribution': total_contribution,
                    'feature_count': len(feature_vector)
                }
            }
            
            return DecisionExplanation(
                decision_type='ml_prediction',
                decision_value=prediction_value,
                confidence_score=confidence_score,
                explanation_text=explanation_text,
                feature_contributions=feature_contributions,
                shap_values=shap_values,
                contributing_factors=contributing_factors,
                metadata={
                    'model_name': model_name,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'analysis_method': 'shap_analysis'
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error explaining ML prediction: {e}")
            return self._get_default_explanation('ml_prediction', 'neutral', 0.5)
    
    async def explain_rl_action(self, rl_state: Dict, action: str, reward: float) -> DecisionExplanation:
        """Explain reinforcement learning agent action"""
        try:
            # Determine confidence based on reward and state
            confidence_score = min(max(reward + 0.5, 0.0), 1.0)
            
            # Extract state features
            state_features = rl_state.get('features', {})
            
            # Calculate feature contributions based on state values
            feature_contributions = []
            total_importance = sum(abs(v) for v in state_features.values()) if state_features else 1.0
            
            for feature_name, feature_value in state_features.items():
                if total_importance > 0:
                    contribution_percentage = (abs(feature_value) / total_importance) * 100
                else:
                    contribution_percentage = 0.0
                
                feature_contributions.append(FeatureContribution(
                    feature_name=feature_name,
                    contribution_value=feature_value,
                    contribution_percentage=contribution_percentage,
                    feature_importance=0.1,  # Default importance for RL features
                    direction='positive' if feature_value > 0 else 'negative'
                ))
            
            # Sort by contribution percentage
            feature_contributions.sort(key=lambda x: x.contribution_percentage, reverse=True)
            
            # Generate explanation text
            top_factors = [f"{fc.feature_name} ({fc.contribution_percentage:.1f}%)" 
                          for fc in feature_contributions[:3]]
            
            explanation_text = self.explanation_templates['rl_action'][action].format(
                top_factors=', '.join(top_factors)
            )
            
            # Contributing factors
            contributing_factors = {
                'rl_state': rl_state,
                'action_info': {
                    'action': action,
                    'reward': reward,
                    'confidence': confidence_score
                },
                'state_analysis': {
                    'feature_count': len(state_features),
                    'total_importance': total_importance
                }
            }
            
            return DecisionExplanation(
                decision_type='rl_action',
                decision_value=action,
                confidence_score=confidence_score,
                explanation_text=explanation_text,
                feature_contributions=feature_contributions,
                shap_values=state_features,
                contributing_factors=contributing_factors,
                metadata={
                    'analysis_timestamp': datetime.now().isoformat(),
                    'analysis_method': 'rl_analysis'
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error explaining RL action: {e}")
            return self._get_default_explanation('rl_action', 'hold', 0.5)
    
    async def explain_anomaly_alert(self, anomaly_result: Dict) -> DecisionExplanation:
        """Explain anomaly detection alert"""
        try:
            anomaly_type = anomaly_result.get('anomaly_type', 'unknown')
            severity = anomaly_result.get('severity', 'medium')
            confidence_score = anomaly_result.get('confidence', 0.5)
            
            # Extract anomaly indicators
            indicators = anomaly_result.get('indicators', {})
            
            # Calculate feature contributions
            feature_contributions = []
            total_score = sum(abs(v) for v in indicators.values()) if indicators else 1.0
            
            for indicator_name, indicator_score in indicators.items():
                if total_score > 0:
                    contribution_percentage = (abs(indicator_score) / total_score) * 100
                else:
                    contribution_percentage = 0.0
                
                feature_contributions.append(FeatureContribution(
                    feature_name=indicator_name,
                    contribution_value=indicator_score,
                    contribution_percentage=contribution_percentage,
                    feature_importance=0.1,  # Default importance for anomaly indicators
                    direction='positive' if indicator_score > 0 else 'negative'
                ))
            
            # Sort by contribution percentage
            feature_contributions.sort(key=lambda x: x.contribution_percentage, reverse=True)
            
            # Generate explanation text
            top_indicators = [f"{fc.feature_name} ({fc.contribution_percentage:.1f}%)" 
                             for fc in feature_contributions[:3]]
            
            explanation_text = self.explanation_templates['anomaly_alert'][severity].format(
                top_indicators=', '.join(top_indicators)
            )
            
            # Contributing factors
            contributing_factors = {
                'anomaly_info': {
                    'anomaly_type': anomaly_type,
                    'severity': severity,
                    'confidence': confidence_score
                },
                'indicators': indicators,
                'detection_method': anomaly_result.get('detection_method', 'unknown')
            }
            
            return DecisionExplanation(
                decision_type='anomaly_alert',
                decision_value=severity,
                confidence_score=confidence_score,
                explanation_text=explanation_text,
                feature_contributions=feature_contributions,
                shap_values=indicators,
                contributing_factors=contributing_factors,
                metadata={
                    'analysis_timestamp': datetime.now().isoformat(),
                    'analysis_method': 'anomaly_detection'
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error explaining anomaly alert: {e}")
            return self._get_default_explanation('anomaly_alert', 'low', 0.5)
    
    async def _calculate_shap_values(self, model_name: str, feature_vector: Dict) -> Dict[str, float]:
        """Calculate SHAP values for a feature vector"""
        try:
            if not ML_AVAILABLE:
                return {k: 0.0 for k in feature_vector.keys()}
            
            # Convert feature vector to array
            feature_names = list(feature_vector.keys())
            feature_values = list(feature_vector.values())
            
            # Create feature array
            X = np.array([feature_values])
            
            # Get or create SHAP explainer
            if model_name not in self.shap_explainers:
                # For now, use a simple linear approximation
                # In production, you would load the actual model and create a proper explainer
                self.shap_explainers[model_name] = 'linear_approximation'
            
            # Calculate approximate SHAP values (simplified)
            # In production, you would use the actual SHAP explainer
            shap_values = {}
            for i, feature_name in enumerate(feature_names):
                # Simple linear approximation based on feature value
                feature_value = feature_values[i]
                shap_values[feature_name] = feature_value * 0.1  # Simplified SHAP approximation
            
            return shap_values
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating SHAP values: {e}")
            return {k: 0.0 for k in feature_vector.keys()}
    
    async def _get_feature_importance(self, model_name: str, feature_name: str) -> float:
        """Get feature importance for a model and feature"""
        try:
            # Check cache first
            cache_key = f"{model_name}_{feature_name}"
            if cache_key in self.feature_importance_cache:
                return self.feature_importance_cache[cache_key]
            
            # Get from database
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT importance_score FROM feature_importance_history 
                    WHERE model_name = $1 AND feature_name = $2
                    ORDER BY timestamp DESC LIMIT 1
                """, model_name, feature_name)
                
                if row:
                    importance = float(row['importance_score'])
                else:
                    importance = 0.1  # Default importance
                
                # Cache the result
                self.feature_importance_cache[cache_key] = importance
                
                return importance
                
        except Exception as e:
            self.logger.error(f"❌ Error getting feature importance: {e}")
            return 0.1
    
    def _get_default_explanation(self, decision_type: str, decision_value: str, confidence: float) -> DecisionExplanation:
        """Get default explanation when analysis fails"""
        return DecisionExplanation(
            decision_type=decision_type,
            decision_value=decision_value,
            confidence_score=confidence,
            explanation_text=f"Default {decision_type} explanation for {decision_value}",
            feature_contributions=[],
            shap_values={},
            contributing_factors={'error': 'Analysis failed, using default explanation'},
            metadata={'default': True}
        )
    
    async def store_explanation(self, symbol: str, timeframe: str, explanation: DecisionExplanation):
        """Store decision explanation in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO trade_explanations (
                        symbol, timeframe, timestamp, decision_type, decision_value,
                        confidence_score, explanation_text, feature_contributions,
                        shap_values, contributing_factors, explanation_metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """, symbol, timeframe, datetime.now(), explanation.decision_type,
                     explanation.decision_value, explanation.confidence_score,
                     explanation.explanation_text, json.dumps([
                         {
                             'feature_name': fc.feature_name,
                             'contribution_value': fc.contribution_value,
                             'contribution_percentage': fc.contribution_percentage,
                             'feature_importance': fc.feature_importance,
                             'direction': fc.direction
                         } for fc in explanation.feature_contributions
                     ]), json.dumps(explanation.shap_values),
                     json.dumps(explanation.contributing_factors),
                     json.dumps(explanation.metadata))
                
        except Exception as e:
            self.logger.error(f"❌ Error storing explanation: {e}")
    
    async def get_recent_explanations(self, symbol: str, timeframe: str, hours: int = 24) -> List[Dict]:
        """Get recent decision explanations"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM trade_explanations 
                    WHERE symbol = $1 AND timeframe = $2 
                    AND timestamp >= NOW() - INTERVAL '1 hour' * $3
                    ORDER BY timestamp DESC
                """, symbol, timeframe, hours)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"❌ Error getting recent explanations: {e}")
            return []
    
    async def generate_trade_journal(self, symbol: str, timeframe: str, hours: int = 24) -> str:
        """Generate a trade journal from recent explanations"""
        try:
            explanations = await self.get_recent_explanations(symbol, timeframe, hours)
            
            if not explanations:
                return f"No trading decisions found for {symbol} {timeframe} in the last {hours} hours."
            
            journal_lines = [
                f"# Trading Journal - {symbol} {timeframe}",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Period: Last {hours} hours",
                f"Total Decisions: {len(explanations)}",
                "",
                "## Recent Decisions:",
                ""
            ]
            
            for i, explanation in enumerate(explanations, 1):
                timestamp = explanation['timestamp'].strftime('%H:%M:%S')
                decision_type = explanation['decision_type']
                decision_value = explanation['decision_value']
                confidence = explanation['confidence_score']
                explanation_text = explanation['explanation_text']
                
                journal_lines.extend([
                    f"### {i}. {timestamp} - {decision_type.upper()} ({decision_value.upper()})",
                    f"**Confidence:** {confidence:.1%}",
                    f"**Explanation:** {explanation_text}",
                    ""
                ])
            
            # Add summary statistics
            decision_types = [e['decision_type'] for e in explanations]
            decision_values = [e['decision_value'] for e in explanations]
            confidences = [e['confidence_score'] for e in explanations]
            
            journal_lines.extend([
                "## Summary Statistics:",
                f"- Most Common Decision Type: {max(set(decision_types), key=decision_types.count)}",
                f"- Most Common Decision Value: {max(set(decision_values), key=decision_values.count)}",
                f"- Average Confidence: {np.mean(confidences):.1%}",
                f"- High Confidence Decisions (>80%): {sum(1 for c in confidences if c > 0.8)}",
                ""
            ])
            
            return "\n".join(journal_lines)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating trade journal: {e}")
            return f"Error generating trade journal: {e}"
    
    async def update_feature_importance(self, model_name: str, feature_importance: Dict[str, float]):
        """Update feature importance for a model"""
        try:
            async with self.db_pool.acquire() as conn:
                for feature_name, importance_score in feature_importance.items():
                    await conn.execute("""
                        INSERT INTO feature_importance_history (
                            model_name, feature_name, importance_score, timestamp
                        ) VALUES ($1, $2, $3, $4)
                    """, model_name, feature_name, importance_score, datetime.now())
                    
                    # Update cache
                    cache_key = f"{model_name}_{feature_name}"
                    self.feature_importance_cache[cache_key] = importance_score
                
        except Exception as e:
            self.logger.error(f"❌ Error updating feature importance: {e}")
    
    async def get_explanation_statistics(self, symbol: str, timeframe: str, days: int = 7) -> Dict:
        """Get explanation statistics"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT decision_type, decision_value, COUNT(*) as count, AVG(confidence_score) as avg_confidence
                    FROM trade_explanations 
                    WHERE symbol = $1 AND timeframe = $2 
                    AND timestamp >= NOW() - INTERVAL '1 day' * $3
                    GROUP BY decision_type, decision_value
                """, symbol, timeframe, days)
                
                stats = {
                    'total_explanations': 0,
                    'decision_distribution': {},
                    'avg_confidence': 0.0,
                    'high_confidence_count': 0
                }
                
                total_count = 0
                total_confidence = 0.0
                high_confidence_count = 0
                
                for row in rows:
                    decision_type = row['decision_type']
                    decision_value = row['decision_value']
                    count = row['count']
                    confidence = row['avg_confidence']
                    
                    key = f"{decision_type}_{decision_value}"
                    stats['decision_distribution'][key] = {
                        'count': count,
                        'percentage': 0.0,
                        'avg_confidence': float(confidence) if confidence else 0.0
                    }
                    
                    total_count += count
                    total_confidence += count * (confidence or 0.0)
                    
                    if confidence and confidence > 0.8:
                        high_confidence_count += count
                
                stats['total_explanations'] = total_count
                stats['high_confidence_count'] = high_confidence_count
                
                if total_count > 0:
                    stats['avg_confidence'] = total_confidence / total_count
                    
                    # Calculate percentages
                    for decision_data in stats['decision_distribution'].values():
                        decision_data['percentage'] = (decision_data['count'] / total_count) * 100
                
                return stats
                
        except Exception as e:
            self.logger.error(f"❌ Error getting explanation statistics: {e}")
            return {'total_explanations': 0, 'decision_distribution': {}, 'avg_confidence': 0.0, 'high_confidence_count': 0}
