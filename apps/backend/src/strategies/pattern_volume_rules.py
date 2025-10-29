#!/usr/bin/env python3
"""
Pattern-Specific Volume Rules for AlphaPulse
Configuration system for pattern-specific volume confirmation rules
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class VolumeRuleType(Enum):
    """Types of volume rules"""
    MINIMUM_VOLUME_RATIO = "minimum_volume_ratio"
    VOLUME_SPIKE_REQUIRED = "volume_spike_required"
    VOLUME_TREND_ALIGNMENT = "volume_trend_alignment"
    VOLUME_DIVERGENCE_PENALTY = "volume_divergence_penalty"
    VOLUME_CONSISTENCY = "volume_consistency"
    VOLUME_BREAKOUT_CONFIRMATION = "volume_breakout_confirmation"

class VolumeConfirmationLevel(Enum):
    """Levels of volume confirmation"""
    REQUIRED = "required"
    PREFERRED = "preferred"
    OPTIONAL = "optional"
    PENALTY = "penalty"

@dataclass
class VolumeRule:
    """Individual volume rule configuration"""
    rule_type: VolumeRuleType
    pattern_name: str
    confirmation_level: VolumeConfirmationLevel
    threshold: float
    multiplier: float  # Confidence multiplier
    description: str
    time_window: int = 20  # Analysis window in periods
    enabled: bool = True

@dataclass
class PatternVolumeConfig:
    """Complete volume configuration for a pattern"""
    pattern_name: str
    rules: List[VolumeRule]
    base_confidence_multiplier: float
    description: str
    pattern_category: str  # "reversal", "continuation", "neutral"
    volume_importance: str  # "critical", "important", "moderate", "low"

class PatternVolumeRulesManager:
    """
    Manager for pattern-specific volume rules configuration
    """
    
    def __init__(self):
        self.pattern_configs: Dict[str, PatternVolumeConfig] = {}
        self._initialize_default_rules()
        logger.info("🚀 Pattern Volume Rules Manager initialized")
    
    def _initialize_default_rules(self):
        """
        Initialize default pattern-specific volume rules
        """
        
        # Hammer Pattern Rules
        hammer_rules = [
            VolumeRule(
                rule_type=VolumeRuleType.MINIMUM_VOLUME_RATIO,
                pattern_name="hammer",
                confirmation_level=VolumeConfirmationLevel.PREFERRED,
                threshold=1.2,
                multiplier=1.15,
                description="Hammer requires above-average volume for confirmation",
                time_window=10
            ),
            VolumeRule(
                rule_type=VolumeRuleType.VOLUME_TREND_ALIGNMENT,
                pattern_name="hammer",
                confirmation_level=VolumeConfirmationLevel.PREFERRED,
                threshold=0.1,
                multiplier=1.1,
                description="Volume should align with bullish reversal",
                time_window=15
            ),
            VolumeRule(
                rule_type=VolumeRuleType.VOLUME_DIVERGENCE_PENALTY,
                pattern_name="hammer",
                confirmation_level=VolumeConfirmationLevel.PENALTY,
                threshold=0.3,
                multiplier=0.9,
                description="Volume divergence reduces hammer reliability",
                time_window=20
            )
        ]
        
        self.pattern_configs["hammer"] = PatternVolumeConfig(
            pattern_name="hammer",
            rules=hammer_rules,
            base_confidence_multiplier=1.1,
            description="Bullish reversal pattern requiring volume confirmation",
            pattern_category="reversal",
            volume_importance="important"
        )
        
        # Shooting Star Pattern Rules
        shooting_star_rules = [
            VolumeRule(
                rule_type=VolumeRuleType.MINIMUM_VOLUME_RATIO,
                pattern_name="shooting_star",
                confirmation_level=VolumeConfirmationLevel.REQUIRED,
                threshold=1.3,
                multiplier=1.2,
                description="Shooting star needs high volume for bearish confirmation",
                time_window=10
            ),
            VolumeRule(
                rule_type=VolumeRuleType.VOLUME_SPIKE_REQUIRED,
                pattern_name="shooting_star",
                confirmation_level=VolumeConfirmationLevel.PREFERRED,
                threshold=1.5,
                multiplier=1.15,
                description="Volume spike confirms bearish reversal",
                time_window=5
            ),
            VolumeRule(
                rule_type=VolumeRuleType.VOLUME_DIVERGENCE_PENALTY,
                pattern_name="shooting_star",
                confirmation_level=VolumeConfirmationLevel.PENALTY,
                threshold=0.2,
                multiplier=0.85,
                description="Low volume reduces shooting star reliability",
                time_window=15
            )
        ]
        
        self.pattern_configs["shooting_star"] = PatternVolumeConfig(
            pattern_name="shooting_star",
            rules=shooting_star_rules,
            base_confidence_multiplier=1.15,
            description="Bearish reversal pattern requiring strong volume",
            pattern_category="reversal",
            volume_importance="critical"
        )
        
        # Bullish Engulfing Pattern Rules
        bullish_engulfing_rules = [
            VolumeRule(
                rule_type=VolumeRuleType.MINIMUM_VOLUME_RATIO,
                pattern_name="bullish_engulfing",
                confirmation_level=VolumeConfirmationLevel.REQUIRED,
                threshold=1.5,
                multiplier=1.25,
                description="Bullish engulfing requires strong volume confirmation",
                time_window=10
            ),
            VolumeRule(
                rule_type=VolumeRuleType.VOLUME_BREAKOUT_CONFIRMATION,
                pattern_name="bullish_engulfing",
                confirmation_level=VolumeConfirmationLevel.PREFERRED,
                threshold=2.0,
                multiplier=1.3,
                description="Volume breakout confirms strong bullish signal",
                time_window=5
            ),
            VolumeRule(
                rule_type=VolumeRuleType.VOLUME_CONSISTENCY,
                pattern_name="bullish_engulfing",
                confirmation_level=VolumeConfirmationLevel.OPTIONAL,
                threshold=0.7,
                multiplier=1.05,
                description="Volume consistency adds reliability",
                time_window=20
            )
        ]
        
        self.pattern_configs["bullish_engulfing"] = PatternVolumeConfig(
            pattern_name="bullish_engulfing",
            rules=bullish_engulfing_rules,
            base_confidence_multiplier=1.2,
            description="Strong bullish reversal requiring volume confirmation",
            pattern_category="reversal",
            volume_importance="critical"
        )
        
        # Bearish Engulfing Pattern Rules
        bearish_engulfing_rules = [
            VolumeRule(
                rule_type=VolumeRuleType.MINIMUM_VOLUME_RATIO,
                pattern_name="bearish_engulfing",
                confirmation_level=VolumeConfirmationLevel.REQUIRED,
                threshold=1.5,
                multiplier=1.25,
                description="Bearish engulfing requires strong volume confirmation",
                time_window=10
            ),
            VolumeRule(
                rule_type=VolumeRuleType.VOLUME_BREAKOUT_CONFIRMATION,
                pattern_name="bearish_engulfing",
                confirmation_level=VolumeConfirmationLevel.PREFERRED,
                threshold=2.0,
                multiplier=1.3,
                description="Volume breakout confirms strong bearish signal",
                time_window=5
            ),
            VolumeRule(
                rule_type=VolumeRuleType.VOLUME_CONSISTENCY,
                pattern_name="bearish_engulfing",
                confirmation_level=VolumeConfirmationLevel.OPTIONAL,
                threshold=0.7,
                multiplier=1.05,
                description="Volume consistency adds reliability",
                time_window=20
            )
        ]
        
        self.pattern_configs["bearish_engulfing"] = PatternVolumeConfig(
            pattern_name="bearish_engulfing",
            rules=bearish_engulfing_rules,
            base_confidence_multiplier=1.2,
            description="Strong bearish reversal requiring volume confirmation",
            pattern_category="reversal",
            volume_importance="critical"
        )
        
        # Doji Pattern Rules
        doji_rules = [
            VolumeRule(
                rule_type=VolumeRuleType.MINIMUM_VOLUME_RATIO,
                pattern_name="doji",
                confirmation_level=VolumeConfirmationLevel.OPTIONAL,
                threshold=1.1,
                multiplier=1.1,
                description="Doji benefits from moderate volume confirmation",
                time_window=15
            ),
            VolumeRule(
                rule_type=VolumeRuleType.VOLUME_TREND_ALIGNMENT,
                pattern_name="doji",
                confirmation_level=VolumeConfirmationLevel.OPTIONAL,
                threshold=0.05,
                multiplier=1.05,
                description="Volume trend alignment adds context",
                time_window=20
            ),
            VolumeRule(
                rule_type=VolumeRuleType.VOLUME_DIVERGENCE_PENALTY,
                pattern_name="doji",
                confirmation_level=VolumeConfirmationLevel.PENALTY,
                threshold=0.4,
                multiplier=0.95,
                description="Strong volume divergence reduces doji reliability",
                time_window=15
            )
        ]
        
        self.pattern_configs["doji"] = PatternVolumeConfig(
            pattern_name="doji",
            rules=doji_rules,
            base_confidence_multiplier=1.0,
            description="Neutral pattern with optional volume confirmation",
            pattern_category="neutral",
            volume_importance="moderate"
        )
        
        # Morning Star Pattern Rules
        morning_star_rules = [
            VolumeRule(
                rule_type=VolumeRuleType.MINIMUM_VOLUME_RATIO,
                pattern_name="morning_star",
                confirmation_level=VolumeConfirmationLevel.PREFERRED,
                threshold=1.4,
                multiplier=1.2,
                description="Morning star needs good volume on third candle",
                time_window=15
            ),
            VolumeRule(
                rule_type=VolumeRuleType.VOLUME_TREND_ALIGNMENT,
                pattern_name="morning_star",
                confirmation_level=VolumeConfirmationLevel.PREFERRED,
                threshold=0.15,
                multiplier=1.15,
                description="Volume should align with bullish reversal",
                time_window=20
            ),
            VolumeRule(
                rule_type=VolumeRuleType.VOLUME_CONSISTENCY,
                pattern_name="morning_star",
                confirmation_level=VolumeConfirmationLevel.OPTIONAL,
                threshold=0.6,
                multiplier=1.05,
                description="Volume consistency across pattern",
                time_window=15
            )
        ]
        
        self.pattern_configs["morning_star"] = PatternVolumeConfig(
            pattern_name="morning_star",
            rules=morning_star_rules,
            base_confidence_multiplier=1.15,
            description="Strong bullish reversal pattern",
            pattern_category="reversal",
            volume_importance="important"
        )
        
        # Evening Star Pattern Rules
        evening_star_rules = [
            VolumeRule(
                rule_type=VolumeRuleType.MINIMUM_VOLUME_RATIO,
                pattern_name="evening_star",
                confirmation_level=VolumeConfirmationLevel.PREFERRED,
                threshold=1.4,
                multiplier=1.2,
                description="Evening star needs good volume on third candle",
                time_window=15
            ),
            VolumeRule(
                rule_type=VolumeRuleType.VOLUME_TREND_ALIGNMENT,
                pattern_name="evening_star",
                confirmation_level=VolumeConfirmationLevel.PREFERRED,
                threshold=0.15,
                multiplier=1.15,
                description="Volume should align with bearish reversal",
                time_window=20
            ),
            VolumeRule(
                rule_type=VolumeRuleType.VOLUME_CONSISTENCY,
                pattern_name="evening_star",
                confirmation_level=VolumeConfirmationLevel.OPTIONAL,
                threshold=0.6,
                multiplier=1.05,
                description="Volume consistency across pattern",
                time_window=15
            )
        ]
        
        self.pattern_configs["evening_star"] = PatternVolumeConfig(
            pattern_name="evening_star",
            rules=evening_star_rules,
            base_confidence_multiplier=1.15,
            description="Strong bearish reversal pattern",
            pattern_category="reversal",
            volume_importance="important"
        )
        
        # Breakout Pattern Rules (Generic)
        breakout_rules = [
            VolumeRule(
                rule_type=VolumeRuleType.MINIMUM_VOLUME_RATIO,
                pattern_name="breakout",
                confirmation_level=VolumeConfirmationLevel.REQUIRED,
                threshold=2.0,
                multiplier=1.3,
                description="Breakout patterns require very high volume confirmation",
                time_window=10
            ),
            VolumeRule(
                rule_type=VolumeRuleType.VOLUME_BREAKOUT_CONFIRMATION,
                pattern_name="breakout",
                confirmation_level=VolumeConfirmationLevel.REQUIRED,
                threshold=2.5,
                multiplier=1.4,
                description="Volume breakout confirms strong breakout signal",
                time_window=5
            ),
            VolumeRule(
                rule_type=VolumeRuleType.VOLUME_CONSISTENCY,
                pattern_name="breakout",
                confirmation_level=VolumeConfirmationLevel.PREFERRED,
                threshold=0.8,
                multiplier=1.1,
                description="Volume consistency validates breakout",
                time_window=15
            )
        ]
        
        self.pattern_configs["breakout"] = PatternVolumeConfig(
            pattern_name="breakout",
            rules=breakout_rules,
            base_confidence_multiplier=1.25,
            description="Breakout pattern requiring maximum volume confirmation",
            pattern_category="continuation",
            volume_importance="critical"
        )
        
        # Continuation Pattern Rules (Generic)
        continuation_rules = [
            VolumeRule(
                rule_type=VolumeRuleType.MINIMUM_VOLUME_RATIO,
                pattern_name="continuation",
                confirmation_level=VolumeConfirmationLevel.PREFERRED,
                threshold=1.2,
                multiplier=1.1,
                description="Continuation patterns benefit from volume confirmation",
                time_window=15
            ),
            VolumeRule(
                rule_type=VolumeRuleType.VOLUME_TREND_ALIGNMENT,
                pattern_name="continuation",
                confirmation_level=VolumeConfirmationLevel.PREFERRED,
                threshold=0.1,
                multiplier=1.05,
                description="Volume should align with trend direction",
                time_window=20
            )
        ]
        
        self.pattern_configs["continuation"] = PatternVolumeConfig(
            pattern_name="continuation",
            rules=continuation_rules,
            base_confidence_multiplier=1.05,
            description="Continuation pattern with moderate volume requirements",
            pattern_category="continuation",
            volume_importance="moderate"
        )
    
    def get_pattern_config(self, pattern_name: str) -> Optional[PatternVolumeConfig]:
        """
        Get volume configuration for a specific pattern
        """
        # Try exact match first
        if pattern_name in self.pattern_configs:
            return self.pattern_configs[pattern_name]
        
        # Try pattern category matching
        pattern_lower = pattern_name.lower()
        
        # Check for reversal patterns
        if any(keyword in pattern_lower for keyword in ['reversal', 'engulfing', 'star', 'hammer']):
            if 'bullish' in pattern_lower:
                return self.pattern_configs.get("bullish_engulfing")
            elif 'bearish' in pattern_lower:
                return self.pattern_configs.get("bearish_engulfing")
            else:
                return self.pattern_configs.get("hammer")  # Default reversal
        
        # Check for continuation patterns
        if any(keyword in pattern_lower for keyword in ['continuation', 'breakout', 'flag', 'wedge']):
            return self.pattern_configs.get("continuation")
        
        # Check for neutral patterns
        if any(keyword in pattern_lower for keyword in ['doji', 'neutral', 'indecision']):
            return self.pattern_configs.get("doji")
        
        # Default to continuation pattern
        return self.pattern_configs.get("continuation")
    
    def get_volume_rules(self, pattern_name: str) -> List[VolumeRule]:
        """
        Get volume rules for a specific pattern
        """
        config = self.get_pattern_config(pattern_name)
        if config:
            return config.rules
        return []
    
    def calculate_volume_confidence(
        self, 
        pattern_name: str, 
        volume_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate volume confidence based on pattern-specific rules
        """
        try:
            config = self.get_pattern_config(pattern_name)
            if not config:
                return {
                    'confidence_multiplier': 1.0,
                    'volume_confirmed': False,
                    'rule_evaluations': [],
                    'overall_score': 0.0
                }
            
            rule_evaluations = []
            total_score = 0.0
            total_weight = 0.0
            
            for rule in config.rules:
                if not rule.enabled:
                    continue
                
                # Evaluate rule based on type
                evaluation = self._evaluate_volume_rule(rule, volume_metrics)
                rule_evaluations.append(evaluation)
                
                # Calculate weighted score
                weight = self._get_rule_weight(rule.confirmation_level)
                score = evaluation['score'] * weight
                
                total_score += score
                total_weight += weight
            
            # Calculate overall confidence
            overall_score = total_score / total_weight if total_weight > 0 else 0.0
            confidence_multiplier = config.base_confidence_multiplier * (1.0 + overall_score)
            
            # Determine if volume is confirmed
            volume_confirmed = overall_score > 0.3  # 30% threshold
            
            return {
                'confidence_multiplier': confidence_multiplier,
                'volume_confirmed': volume_confirmed,
                'rule_evaluations': rule_evaluations,
                'overall_score': overall_score,
                'pattern_config': config
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume confidence: {e}")
            return {
                'confidence_multiplier': 1.0,
                'volume_confirmed': False,
                'rule_evaluations': [],
                'overall_score': 0.0
            }
    
    def _evaluate_volume_rule(self, rule: VolumeRule, volume_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a specific volume rule against metrics
        """
        try:
            score = 0.0
            passed = False
            actual_value = 0.0
            
            if rule.rule_type == VolumeRuleType.MINIMUM_VOLUME_RATIO:
                actual_value = volume_metrics.get('volume_ratio', 1.0)
                passed = actual_value >= rule.threshold
                score = min(1.0, actual_value / rule.threshold) if passed else 0.0
                
            elif rule.rule_type == VolumeRuleType.VOLUME_SPIKE_REQUIRED:
                actual_value = volume_metrics.get('volume_spike_ratio', 1.0)
                passed = actual_value >= rule.threshold
                score = min(1.0, actual_value / rule.threshold) if passed else 0.0
                
            elif rule.rule_type == VolumeRuleType.VOLUME_TREND_ALIGNMENT:
                actual_value = volume_metrics.get('volume_trend_alignment', 0.0)
                passed = abs(actual_value) >= rule.threshold
                score = min(1.0, abs(actual_value) / rule.threshold) if passed else 0.0
                
            elif rule.rule_type == VolumeRuleType.VOLUME_DIVERGENCE_PENALTY:
                actual_value = volume_metrics.get('divergence_strength', 0.0)
                passed = actual_value <= rule.threshold  # Lower is better for penalty
                score = max(0.0, 1.0 - (actual_value / rule.threshold)) if passed else 0.0
                
            elif rule.rule_type == VolumeRuleType.VOLUME_CONSISTENCY:
                actual_value = volume_metrics.get('volume_consistency', 0.0)
                passed = actual_value >= rule.threshold
                score = actual_value if passed else 0.0
                
            elif rule.rule_type == VolumeRuleType.VOLUME_BREAKOUT_CONFIRMATION:
                actual_value = volume_metrics.get('breakout_volume_ratio', 1.0)
                passed = actual_value >= rule.threshold
                score = min(1.0, actual_value / rule.threshold) if passed else 0.0
            
            return {
                'rule_type': rule.rule_type.value,
                'pattern_name': rule.pattern_name,
                'confirmation_level': rule.confirmation_level.value,
                'threshold': rule.threshold,
                'actual_value': actual_value,
                'passed': passed,
                'score': score,
                'multiplier': rule.multiplier,
                'description': rule.description
            }
            
        except Exception as e:
            logger.error(f"Error evaluating volume rule: {e}")
            return {
                'rule_type': rule.rule_type.value,
                'pattern_name': rule.pattern_name,
                'confirmation_level': rule.confirmation_level.value,
                'threshold': rule.threshold,
                'actual_value': 0.0,
                'passed': False,
                'score': 0.0,
                'multiplier': 1.0,
                'description': rule.description
            }
    
    def _get_rule_weight(self, confirmation_level: VolumeConfirmationLevel) -> float:
        """
        Get weight for rule based on confirmation level
        """
        weights = {
            VolumeConfirmationLevel.REQUIRED: 1.0,
            VolumeConfirmationLevel.PREFERRED: 0.7,
            VolumeConfirmationLevel.OPTIONAL: 0.3,
            VolumeConfirmationLevel.PENALTY: 0.5
        }
        return weights.get(confirmation_level, 0.5)
    
    def add_custom_rule(self, rule: VolumeRule):
        """
        Add a custom volume rule
        """
        try:
            pattern_name = rule.pattern_name
            
            if pattern_name not in self.pattern_configs:
                # Create new pattern config
                self.pattern_configs[pattern_name] = PatternVolumeConfig(
                    pattern_name=pattern_name,
                    rules=[rule],
                    base_confidence_multiplier=1.0,
                    description=f"Custom pattern: {pattern_name}",
                    pattern_category="custom",
                    volume_importance="moderate"
                )
            else:
                # Add rule to existing config
                self.pattern_configs[pattern_name].rules.append(rule)
            
            logger.info(f"✅ Added custom volume rule for pattern: {pattern_name}")
            
        except Exception as e:
            logger.error(f"Error adding custom rule: {e}")
    
    def update_rule(self, pattern_name: str, rule_type: VolumeRuleType, **updates):
        """
        Update an existing volume rule
        """
        try:
            config = self.get_pattern_config(pattern_name)
            if not config:
                logger.warning(f"Pattern config not found: {pattern_name}")
                return
            
            for rule in config.rules:
                if rule.rule_type == rule_type:
                    for key, value in updates.items():
                        if hasattr(rule, key):
                            setattr(rule, key, value)
                    logger.info(f"✅ Updated rule {rule_type.value} for pattern: {pattern_name}")
                    return
            
            logger.warning(f"Rule {rule_type.value} not found for pattern: {pattern_name}")
            
        except Exception as e:
            logger.error(f"Error updating rule: {e}")
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """
        Get summary of all pattern configurations
        """
        try:
            summary = {
                'total_patterns': len(self.pattern_configs),
                'patterns_by_category': {},
                'patterns_by_importance': {},
                'patterns': []
            }
            
            for pattern_name, config in self.pattern_configs.items():
                # Category grouping
                category = config.pattern_category
                if category not in summary['patterns_by_category']:
                    summary['patterns_by_category'][category] = []
                summary['patterns_by_category'][category].append(pattern_name)
                
                # Importance grouping
                importance = config.volume_importance
                if importance not in summary['patterns_by_importance']:
                    summary['patterns_by_importance'][importance] = []
                summary['patterns_by_importance'][importance].append(pattern_name)
                
                # Pattern details
                pattern_info = {
                    'name': pattern_name,
                    'category': config.pattern_category,
                    'importance': config.volume_importance,
                    'base_multiplier': config.base_confidence_multiplier,
                    'rule_count': len(config.rules),
                    'description': config.description
                }
                summary['patterns'].append(pattern_info)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting pattern summary: {e}")
            return {'error': str(e)}
