"""
Threshold Environment Management for AlphaPulse

This module provides dynamic threshold management, environment-based configuration,
and adaptive parameter tuning for the trading system.
"""

import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime, timedelta
import asyncio
import logging
from enum import Enum
import redis
from pathlib import Path

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Trading environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    BACKTEST = "backtest"


@dataclass
class ThresholdConfig:
    """Configuration for dynamic thresholds."""
    name: str
    default_value: float
    min_value: float
    max_value: float
    step_size: float = 0.01
    adaptive: bool = True
    environment: Environment = Environment.PRODUCTION
    description: str = ""


@dataclass
class ThresholdState:
    """Current state of a threshold."""
    name: str
    current_value: float
    last_updated: datetime
    performance_score: float = 0.0
    adjustment_count: int = 0
    environment: Environment = Environment.PRODUCTION


class ThresholdManager:
    """
    Dynamic threshold management system for AlphaPulse.
    
    Handles environment-based configuration, adaptive threshold tuning,
    and performance-based parameter optimization.
    """
    
    def __init__(
        self,
        environment: Environment = Environment.PRODUCTION,
        redis_client: Optional[redis.Redis] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the threshold manager.
        
        Args:
            environment: Current trading environment
            redis_client: Redis client for persistence
            config_path: Path to configuration file
        """
        self.environment = environment
        self.redis_client = redis_client
        self.config_path = config_path or "config/thresholds.json"
        
        # Initialize thresholds
        self.thresholds: Dict[str, ThresholdConfig] = {}
        self.threshold_states: Dict[str, ThresholdState] = {}
        self.performance_history: List[Dict] = []
        
        # Load configuration
        self._load_configuration()
        self._initialize_thresholds()
    
    def _load_configuration(self):
        """Load threshold configuration from file."""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                for threshold_data in config_data.get('thresholds', []):
                    config = ThresholdConfig(
                        name=threshold_data['name'],
                        default_value=threshold_data['default_value'],
                        min_value=threshold_data['min_value'],
                        max_value=threshold_data['max_value'],
                        step_size=threshold_data.get('step_size', 0.01),
                        adaptive=threshold_data.get('adaptive', True),
                        environment=Environment(threshold_data.get('environment', 'production')),
                        description=threshold_data.get('description', '')
                    )
                    self.thresholds[config.name] = config
                    
                logger.info(f"Loaded {len(self.thresholds)} threshold configurations")
            else:
                logger.warning(f"Configuration file not found: {self.config_path}")
                self._create_default_configuration()
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._create_default_configuration()
    
    def _create_default_configuration(self):
        """Create default threshold configuration."""
        default_thresholds = [
            {
                'name': 'rsi_oversold',
                'default_value': 30.0,
                'min_value': 20.0,
                'max_value': 40.0,
                'step_size': 1.0,
                'adaptive': True,
                'environment': 'production',
                'description': 'RSI oversold threshold'
            },
            {
                'name': 'rsi_overbought',
                'default_value': 70.0,
                'min_value': 60.0,
                'max_value': 80.0,
                'step_size': 1.0,
                'adaptive': True,
                'environment': 'production',
                'description': 'RSI overbought threshold'
            },
            {
                'name': 'volume_threshold',
                'default_value': 1.5,
                'min_value': 1.0,
                'max_value': 3.0,
                'step_size': 0.1,
                'adaptive': True,
                'environment': 'production',
                'description': 'Volume confirmation threshold'
            },
            {
                'name': 'confidence_threshold',
                'default_value': 0.7,
                'min_value': 0.5,
                'max_value': 0.9,
                'step_size': 0.05,
                'adaptive': True,
                'environment': 'production',
                'description': 'Signal confidence threshold'
            },
            {
                'name': 'adx_threshold',
                'default_value': 25.0,
                'min_value': 20.0,
                'max_value': 35.0,
                'step_size': 1.0,
                'adaptive': True,
                'environment': 'production',
                'description': 'ADX trend strength threshold'
            },
            {
                'name': 'breakout_strength',
                'default_value': 0.7,
                'min_value': 0.5,
                'max_value': 0.9,
                'step_size': 0.05,
                'adaptive': True,
                'environment': 'production',
                'description': 'Breakout strength threshold'
            }
        ]
        
        for threshold_data in default_thresholds:
            config = ThresholdConfig(
                name=threshold_data['name'],
                default_value=threshold_data['default_value'],
                min_value=threshold_data['min_value'],
                max_value=threshold_data['max_value'],
                step_size=threshold_data['step_size'],
                adaptive=threshold_data['adaptive'],
                environment=Environment(threshold_data['environment']),
                description=threshold_data['description']
            )
            self.thresholds[config.name] = config
        
        logger.info("Created default threshold configuration")
    
    def _initialize_thresholds(self):
        """Initialize threshold states."""
        for name, config in self.thresholds.items():
            # Try to load from Redis first
            current_value = self._load_from_redis(name)
            
            if current_value is None:
                current_value = config.default_value
            
            state = ThresholdState(
                name=name,
                current_value=current_value,
                last_updated=datetime.now(),
                environment=self.environment
            )
            self.threshold_states[name] = state
    
    def _load_from_redis(self, threshold_name: str) -> Optional[float]:
        """Load threshold value from Redis."""
        if not self.redis_client:
            return None
        
        try:
            key = f"threshold:{threshold_name}:{self.environment.value}"
            value = self.redis_client.get(key)
            return float(value) if value else None
        except Exception as e:
            logger.warning(f"Error loading from Redis: {e}")
            return None
    
    def _save_to_redis(self, threshold_name: str, value: float):
        """Save threshold value to Redis."""
        if not self.redis_client:
            return
        
        try:
            key = f"threshold:{threshold_name}:{self.environment.value}"
            self.redis_client.set(key, str(value))
        except Exception as e:
            logger.warning(f"Error saving to Redis: {e}")
    
    def get_threshold(self, name: str) -> float:
        """
        Get current threshold value.
        
        Args:
            name: Threshold name
            
        Returns:
            Current threshold value
        """
        if name not in self.threshold_states:
            logger.warning(f"Threshold not found: {name}")
            return 0.0
        
        return self.threshold_states[name].current_value
    
    def set_threshold(self, name: str, value: float, force: bool = False):
        """
        Set threshold value.
        
        Args:
            name: Threshold name
            value: New threshold value
            force: Force update even if not adaptive
        """
        if name not in self.thresholds:
            logger.warning(f"Threshold not found: {name}")
            return
        
        config = self.thresholds[name]
        
        # Validate value range
        if not force and not config.adaptive:
            logger.warning(f"Threshold {name} is not adaptive")
            return
        
        # Clamp value to range
        clamped_value = max(config.min_value, min(config.max_value, value))
        
        if clamped_value != value:
            logger.info(f"Clamped threshold {name} from {value} to {clamped_value}")
        
        # Update state
        state = self.threshold_states[name]
        state.current_value = clamped_value
        state.last_updated = datetime.now()
        state.adjustment_count += 1
        
        # Save to Redis
        self._save_to_redis(name, clamped_value)
        
        logger.info(f"Updated threshold {name} to {clamped_value}")
    
    def adjust_threshold(
        self,
        name: str,
        performance_score: float,
        direction: str = 'auto'
    ):
        """
        Adjust threshold based on performance.
        
        Args:
            name: Threshold name
            performance_score: Performance score (0-1)
            direction: Adjustment direction ('up', 'down', 'auto')
        """
        if name not in self.thresholds:
            logger.warning(f"Threshold not found: {name}")
            return
        
        config = self.thresholds[name]
        state = self.threshold_states[name]
        
        if not config.adaptive:
            logger.warning(f"Threshold {name} is not adaptive")
            return
        
        # Update performance score
        state.performance_score = performance_score
        
        # Determine adjustment direction
        if direction == 'auto':
            if performance_score > 0.8:
                # Good performance, make threshold more conservative
                direction = 'up' if name in ['rsi_oversold', 'volume_threshold', 'confidence_threshold'] else 'down'
            elif performance_score < 0.4:
                # Poor performance, make threshold more aggressive
                direction = 'down' if name in ['rsi_oversold', 'volume_threshold', 'confidence_threshold'] else 'up'
            else:
                return  # No adjustment needed
        
        # Calculate new value
        current_value = state.current_value
        step_size = config.step_size
        
        if direction == 'up':
            new_value = current_value + step_size
        else:
            new_value = current_value - step_size
        
        # Set new threshold
        self.set_threshold(name, new_value)
    
    def get_environment_thresholds(self) -> Dict[str, float]:
        """
        Get all thresholds for current environment.
        
        Returns:
            Dictionary of threshold names and values
        """
        return {
            name: state.current_value
            for name, state in self.threshold_states.items()
            if state.environment == self.environment
        }
    
    def update_performance(self, performance_data: Dict[str, Any]):
        """
        Update performance metrics and adjust thresholds.
        
        Args:
            performance_data: Performance metrics dictionary
        """
        self.performance_history.append({
            'timestamp': datetime.now(),
            'data': performance_data
        })
        
        # Keep only last 100 performance records
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # Adjust thresholds based on performance
        accuracy = performance_data.get('accuracy', 0.5)
        filter_rate = performance_data.get('filter_rate', 0.5)
        
        # Adjust confidence threshold based on accuracy
        if 'confidence_threshold' in self.threshold_states:
            self.adjust_threshold('confidence_threshold', accuracy)
        
        # Adjust volume threshold based on filter rate
        if 'volume_threshold' in self.threshold_states:
            self.adjust_threshold('volume_threshold', filter_rate)
        
        # Adjust RSI thresholds based on overall performance
        overall_score = (accuracy + filter_rate) / 2
        if 'rsi_oversold' in self.threshold_states:
            self.adjust_threshold('rsi_oversold', overall_score)
        if 'rsi_overbought' in self.threshold_states:
            self.adjust_threshold('rsi_overbought', overall_score)
    
    def get_threshold_summary(self) -> Dict[str, Any]:
        """
        Get summary of all thresholds.
        
        Returns:
            Dictionary with threshold summary
        """
        summary = {
            'environment': self.environment.value,
            'total_thresholds': len(self.thresholds),
            'adaptive_thresholds': sum(1 for t in self.thresholds.values() if t.adaptive),
            'thresholds': {}
        }
        
        for name, config in self.thresholds.items():
            state = self.threshold_states[name]
            summary['thresholds'][name] = {
                'current_value': state.current_value,
                'default_value': config.default_value,
                'min_value': config.min_value,
                'max_value': config.max_value,
                'adaptive': config.adaptive,
                'performance_score': state.performance_score,
                'adjustment_count': state.adjustment_count,
                'last_updated': state.last_updated.isoformat(),
                'description': config.description
            }
        
        return summary
    
    def reset_thresholds(self, names: Optional[List[str]] = None):
        """
        Reset thresholds to default values.
        
        Args:
            names: List of threshold names to reset (None for all)
        """
        if names is None:
            names = list(self.thresholds.keys())
        
        for name in names:
            if name in self.thresholds:
                config = self.thresholds[name]
                self.set_threshold(name, config.default_value, force=True)
                logger.info(f"Reset threshold {name} to default value {config.default_value}")
    
    def export_configuration(self, file_path: str):
        """
        Export current configuration to file.
        
        Args:
            file_path: Path to export configuration
        """
        config_data = {
            'environment': self.environment.value,
            'exported_at': datetime.now().isoformat(),
            'thresholds': []
        }
        
        for name, config in self.thresholds.items():
            state = self.threshold_states[name]
            threshold_data = {
                'name': name,
                'default_value': config.default_value,
                'min_value': config.min_value,
                'max_value': config.max_value,
                'step_size': config.step_size,
                'adaptive': config.adaptive,
                'environment': config.environment.value,
                'description': config.description,
                'current_value': state.current_value,
                'performance_score': state.performance_score,
                'adjustment_count': state.adjustment_count
            }
            config_data['thresholds'].append(threshold_data)
        
        try:
            with open(file_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"Exported configuration to {file_path}")
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")


class EnvironmentManager:
    """
    Environment-specific configuration manager.
    """
    
    def __init__(self, base_config: Dict[str, Any]):
        """
        Initialize environment manager.
        
        Args:
            base_config: Base configuration dictionary
        """
        self.base_config = base_config
        self.environments = {
            Environment.DEVELOPMENT: self._get_development_config(),
            Environment.STAGING: self._get_staging_config(),
            Environment.PRODUCTION: self._get_production_config(),
            Environment.BACKTEST: self._get_backtest_config()
        }
    
    def _get_development_config(self) -> Dict[str, Any]:
        """Get development environment configuration."""
        config = self.base_config.copy()
        config.update({
            'debug': True,
            'log_level': 'DEBUG',
            'risk_multiplier': 0.1,  # 10% of production risk
            'max_positions': 2,
            'latency_target': 100,  # 100ms for development
            'use_mock_data': True
        })
        return config
    
    def _get_staging_config(self) -> Dict[str, Any]:
        """Get staging environment configuration."""
        config = self.base_config.copy()
        config.update({
            'debug': False,
            'log_level': 'INFO',
            'risk_multiplier': 0.5,  # 50% of production risk
            'max_positions': 5,
            'latency_target': 75,  # 75ms for staging
            'use_mock_data': False
        })
        return config
    
    def _get_production_config(self) -> Dict[str, Any]:
        """Get production environment configuration."""
        config = self.base_config.copy()
        config.update({
            'debug': False,
            'log_level': 'WARNING',
            'risk_multiplier': 1.0,  # Full production risk
            'max_positions': 10,
            'latency_target': 50,  # 50ms for production
            'use_mock_data': False
        })
        return config
    
    def _get_backtest_config(self) -> Dict[str, Any]:
        """Get backtest environment configuration."""
        config = self.base_config.copy()
        config.update({
            'debug': True,
            'log_level': 'INFO',
            'risk_multiplier': 1.0,  # Full risk for backtesting
            'max_positions': 100,  # No limit for backtesting
            'latency_target': 1000,  # No latency constraint
            'use_mock_data': True
        })
        return config
    
    def get_config(self, environment: Environment) -> Dict[str, Any]:
        """
        Get configuration for specific environment.
        
        Args:
            environment: Target environment
            
        Returns:
            Environment-specific configuration
        """
        return self.environments.get(environment, self.base_config)
    
    def get_current_config(self) -> Dict[str, Any]:
        """
        Get configuration for current environment.
        
        Returns:
            Current environment configuration
        """
        env_name = os.getenv('ALPHAPULSE_ENV', 'production')
        environment = Environment(env_name)
        return self.get_config(environment)


# Example usage
async def example_usage():
    """Example usage of the threshold environment system."""
    
    # Initialize threshold manager
    threshold_manager = ThresholdManager(
        environment=Environment.DEVELOPMENT
    )
    
    # Get threshold values
    rsi_oversold = threshold_manager.get_threshold('rsi_oversold')
    confidence_threshold = threshold_manager.get_threshold('confidence_threshold')
    
    print(f"RSI Oversold: {rsi_oversold}")
    print(f"Confidence Threshold: {confidence_threshold}")
    
    # Update performance and adjust thresholds
    performance_data = {
        'accuracy': 0.75,
        'filter_rate': 0.65,
        'latency_avg': 45.2
    }
    
    threshold_manager.update_performance(performance_data)
    
    # Get updated thresholds
    updated_rsi = threshold_manager.get_threshold('rsi_oversold')
    print(f"Updated RSI Oversold: {updated_rsi}")
    
    # Get summary
    summary = threshold_manager.get_threshold_summary()
    print(f"Threshold Summary: {summary}")


if __name__ == "__main__":
    asyncio.run(example_usage())
