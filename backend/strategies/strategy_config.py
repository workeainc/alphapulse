"""
Strategy Configuration System for AlphaPulse
Week 8: Strategy Configuration and Performance Monitoring

Features:
- Dynamic risk parameter management
- Strategy-specific configuration profiles
- Real-time parameter updates
- Risk validation and enforcement

Author: AlphaPulse Team
Date: 2025
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    """Available strategy types"""
    FUNDING_RATE = "funding_rate"
    CROSS_CHAIN = "cross_chain"
    VOLATILITY = "volatility"
    ARBITRAGE = "arbitrage"
    CORRELATION = "correlation"
    PREDICTIVE = "predictive"

class RiskLevel(Enum):
    """Risk tolerance levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"

@dataclass
class StrategyConfig:
    """Strategy configuration with risk parameters"""
    strategy_id: str
    symbol: Optional[str] = None  # None for global config
    max_loss_pct: float = 0.05  # 5% max daily loss
    take_profit_pct: float = 0.03  # 3% take profit
    stop_loss_pct: float = 0.02  # 2% stop loss
    max_leverage: float = 10.0  # 10x leverage
    position_size_pct: float = 0.1  # 10% of capital per position
    risk_reward_ratio: float = 2.0  # Minimum 2:1 risk/reward
    max_daily_trades: int = 10  # Maximum trades per day
    max_open_positions: int = 5  # Maximum concurrent positions
    correlation_threshold: float = 0.7  # Minimum correlation for signals
    volatility_threshold: float = 0.02  # Minimum volatility for signals
    funding_rate_threshold: float = 0.001  # Minimum funding rate change
    confidence_threshold: float = 0.7  # Minimum signal confidence
    timeout_hours: int = 24  # Position timeout in hours
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate configuration parameters"""
        if self.max_loss_pct <= 0 or self.max_loss_pct > 1:
            raise ValueError("max_loss_pct must be between 0 and 1")
        
        if self.take_profit_pct <= 0 or self.take_profit_pct > 1:
            raise ValueError("take_profit_pct must be between 0 and 1")
        
        if self.stop_loss_pct <= 0 or self.stop_loss_pct > 1:
            raise ValueError("stop_loss_pct must be between 0 and 1")
        
        if self.max_leverage <= 0 or self.max_leverage > 100:
            raise ValueError("max_leverage must be between 0 and 100")
        
        if self.position_size_pct <= 0 or self.position_size_pct > 1:
            raise ValueError("position_size_pct must be between 0 and 1")
        
        if self.risk_reward_ratio <= 0:
            raise ValueError("risk_reward_ratio must be positive")
        
        if self.max_daily_trades <= 0:
            raise ValueError("max_daily_trades must be positive")
        
        if self.max_open_positions <= 0:
            raise ValueError("max_open_positions must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        config_dict = asdict(self)
        config_dict['timestamp'] = datetime.now(timezone.utc)
        return config_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyConfig':
        """Create from dictionary"""
        # Remove timestamp and created_at if present
        config_data = {k: v for k, v in data.items() 
                      if k not in ['timestamp', 'created_at', 'id']}
        return cls(**config_data)
    
    @classmethod
    def get_preset_config(cls, strategy_type: StrategyType, risk_level: RiskLevel, 
                         symbol: str = None) -> 'StrategyConfig':
        """Get preset configuration based on strategy type and risk level"""
        base_configs = {
            StrategyType.FUNDING_RATE: {
                RiskLevel.CONSERVATIVE: {
                    'max_loss_pct': 0.02, 'take_profit_pct': 0.02, 'stop_loss_pct': 0.01,
                    'max_leverage': 5.0, 'position_size_pct': 0.05, 'max_daily_trades': 5
                },
                RiskLevel.MODERATE: {
                    'max_loss_pct': 0.05, 'take_profit_pct': 0.03, 'stop_loss_pct': 0.02,
                    'max_leverage': 10.0, 'position_size_pct': 0.1, 'max_daily_trades': 10
                },
                RiskLevel.AGGRESSIVE: {
                    'max_loss_pct': 0.10, 'take_profit_pct': 0.05, 'stop_loss_pct': 0.03,
                    'max_leverage': 20.0, 'position_size_pct': 0.15, 'max_daily_trades': 15
                },
                RiskLevel.EXTREME: {
                    'max_loss_pct': 0.20, 'take_profit_pct': 0.08, 'stop_loss_pct': 0.05,
                    'max_leverage': 50.0, 'position_size_pct': 0.25, 'max_daily_trades': 25
                }
            },
            StrategyType.CROSS_CHAIN: {
                RiskLevel.CONSERVATIVE: {
                    'max_loss_pct': 0.03, 'take_profit_pct': 0.025, 'stop_loss_pct': 0.015,
                    'max_leverage': 3.0, 'position_size_pct': 0.03, 'max_daily_trades': 3
                },
                RiskLevel.MODERATE: {
                    'max_loss_pct': 0.06, 'take_profit_pct': 0.04, 'stop_loss_pct': 0.025,
                    'max_leverage': 8.0, 'position_size_pct': 0.08, 'max_daily_trades': 8
                },
                RiskLevel.AGGRESSIVE: {
                    'max_loss_pct': 0.12, 'take_profit_pct': 0.06, 'stop_loss_pct': 0.04,
                    'max_leverage': 15.0, 'position_size_pct': 0.12, 'max_daily_trades': 12
                },
                RiskLevel.EXTREME: {
                    'max_loss_pct': 0.25, 'take_profit_pct': 0.10, 'stop_loss_pct': 0.06,
                    'max_leverage': 30.0, 'position_size_pct': 0.20, 'max_daily_trades': 20
                }
            },
            StrategyType.VOLATILITY: {
                RiskLevel.CONSERVATIVE: {
                    'max_loss_pct': 0.025, 'take_profit_pct': 0.02, 'stop_loss_pct': 0.015,
                    'max_leverage': 4.0, 'position_size_pct': 0.04, 'max_daily_trades': 4
                },
                RiskLevel.MODERATE: {
                    'max_loss_pct': 0.055, 'take_profit_pct': 0.035, 'stop_loss_pct': 0.025,
                    'max_leverage': 9.0, 'position_size_pct': 0.09, 'max_daily_trades': 9
                },
                RiskLevel.AGGRESSIVE: {
                    'max_loss_pct': 0.11, 'take_profit_pct': 0.055, 'stop_loss_pct': 0.035,
                    'max_leverage': 18.0, 'position_size_pct': 0.18, 'max_daily_trades': 18
                },
                RiskLevel.EXTREME: {
                    'max_loss_pct': 0.22, 'take_profit_pct': 0.09, 'stop_loss_pct': 0.055,
                    'max_leverage': 45.0, 'position_size_pct': 0.22, 'max_daily_trades': 22
                }
            },
            StrategyType.ARBITRAGE: {
                RiskLevel.CONSERVATIVE: {
                    'max_loss_pct': 0.015, 'take_profit_pct': 0.015, 'stop_loss_pct': 0.01,
                    'max_leverage': 2.0, 'position_size_pct': 0.02, 'max_daily_trades': 2
                },
                RiskLevel.MODERATE: {
                    'max_loss_pct': 0.04, 'take_profit_pct': 0.025, 'stop_loss_pct': 0.02,
                    'max_leverage': 6.0, 'position_size_pct': 0.06, 'max_daily_trades': 6
                },
                RiskLevel.AGGRESSIVE: {
                    'max_loss_pct': 0.08, 'take_profit_pct': 0.04, 'stop_loss_pct': 0.025,
                    'max_leverage': 12.0, 'position_size_pct': 0.12, 'max_daily_trades': 12
                },
                RiskLevel.EXTREME: {
                    'max_loss_pct': 0.18, 'take_profit_pct': 0.07, 'stop_loss_pct': 0.04,
                    'max_leverage': 25.0, 'position_size_pct': 0.18, 'max_daily_trades': 18
                }
            },
            StrategyType.CORRELATION: {
                RiskLevel.CONSERVATIVE: {
                    'max_loss_pct': 0.02, 'take_profit_pct': 0.018, 'stop_loss_pct': 0.012,
                    'max_leverage': 3.5, 'position_size_pct': 0.035, 'max_daily_trades': 3
                },
                RiskLevel.MODERATE: {
                    'max_loss_pct': 0.05, 'take_profit_pct': 0.032, 'stop_loss_pct': 0.022,
                    'max_leverage': 8.5, 'position_size_pct': 0.085, 'max_daily_trades': 8
                },
                RiskLevel.AGGRESSIVE: {
                    'max_loss_pct': 0.105, 'take_profit_pct': 0.052, 'stop_loss_pct': 0.032,
                    'max_leverage': 16.0, 'position_size_pct': 0.16, 'max_daily_trades': 16
                },
                RiskLevel.EXTREME: {
                    'max_loss_pct': 0.21, 'take_profit_pct': 0.085, 'stop_loss_pct': 0.052,
                    'max_leverage': 40.0, 'position_size_pct': 0.21, 'max_daily_trades': 21
                }
            },
            StrategyType.PREDICTIVE: {
                RiskLevel.CONSERVATIVE: {
                    'max_loss_pct': 0.018, 'take_profit_pct': 0.016, 'stop_loss_pct': 0.01,
                    'max_leverage': 3.0, 'position_size_pct': 0.03, 'max_daily_trades': 3
                },
                RiskLevel.MODERATE: {
                    'max_loss_pct': 0.045, 'take_profit_pct': 0.03, 'stop_loss_pct': 0.02,
                    'max_leverage': 7.5, 'position_size_pct': 0.075, 'max_daily_trades': 7
                },
                RiskLevel.AGGRESSIVE: {
                    'max_loss_pct': 0.095, 'take_profit_pct': 0.048, 'stop_loss_pct': 0.03,
                    'max_leverage': 14.0, 'position_size_pct': 0.14, 'max_daily_trades': 14
                },
                RiskLevel.EXTREME: {
                    'max_loss_pct': 0.19, 'take_profit_pct': 0.075, 'stop_loss_pct': 0.048,
                    'max_leverage': 35.0, 'position_size_pct': 0.19, 'max_daily_trades': 19
                }
            }
        }
        
        if strategy_type not in base_configs or risk_level not in base_configs[strategy_type]:
            raise ValueError(f"No preset config for {strategy_type} - {risk_level}")
        
        preset = base_configs[strategy_type][risk_level]
        preset['strategy_id'] = strategy_type.value
        preset['symbol'] = symbol
        
        return cls(**preset)

class StrategyConfigManager:
    """Manages strategy configurations with database persistence"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.configs: Dict[str, StrategyConfig] = {}
        self.logger = logger
        
        # Default configurations
        self._default_configs = {}
        self._initialize_default_configs()
    
    def _initialize_default_configs(self):
        """Initialize default configurations for all strategy types"""
        for strategy_type in StrategyType:
            for risk_level in RiskLevel:
                try:
                    config = StrategyConfig.get_preset_config(strategy_type, risk_level)
                    key = f"{strategy_type.value}_{risk_level.value}"
                    self._default_configs[key] = config
                except Exception as e:
                    self.logger.warning(f"Failed to create default config for {key}: {e}")
    
    async def load_config(self, strategy_id: str, symbol: str = None) -> StrategyConfig:
        """Load configuration from database or return default"""
        try:
            # Try to load from database
            db_config = await self.db.get_strategy_config(strategy_id, symbol)
            
            if db_config:
                config = StrategyConfig.from_dict(db_config)
                self.logger.info(f"Loaded config for {strategy_id} - {symbol} from database")
                return config
            
            # Return default configuration
            default_key = f"{strategy_id}_moderate"
            if default_key in self._default_configs:
                default_config = self._default_configs[default_key]
                if symbol:
                    default_config.symbol = symbol
                self.logger.info(f"Using default config for {strategy_id} - {symbol}")
                return default_config
            
            # Create new default configuration
            config = StrategyConfig(
                strategy_id=strategy_id,
                symbol=symbol
            )
            self.logger.info(f"Created new default config for {strategy_id} - {symbol}")
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading config for {strategy_id} - {symbol}: {e}")
            # Return safe default
            return StrategyConfig(
                strategy_id=strategy_id,
                symbol=symbol,
                max_loss_pct=0.05,
                take_profit_pct=0.03,
                stop_loss_pct=0.02,
                max_leverage=10.0,
                position_size_pct=0.1
            )
    
    async def save_config(self, config: StrategyConfig) -> bool:
        """Save configuration to database"""
        try:
            config_data = config.to_dict()
            success = await self.db.save_strategy_config(config_data)
            
            if success:
                # Update local cache
                key = f"{config.strategy_id}_{config.symbol or 'global'}"
                self.configs[key] = config
                self.logger.info(f"Config saved for {config.strategy_id} - {config.symbol}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            return False
    
    async def update_config(self, strategy_id: str, symbol: str, 
                           updates: Dict[str, Any]) -> bool:
        """Update existing configuration"""
        try:
            # Load current config
            current_config = await self.load_config(strategy_id, symbol)
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(current_config, key):
                    setattr(current_config, key, value)
            
            # Validate updated config
            current_config._validate_parameters()
            
            # Save to database
            success = await self.db.update_strategy_config(current_config.to_dict())
            
            if success:
                # Update local cache
                key = f"{strategy_id}_{symbol or 'global'}"
                self.configs[key] = current_config
                self.logger.info(f"Config updated for {strategy_id} - {symbol}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error updating config: {e}")
            return False
    
    async def get_all_configs(self, strategy_id: str = None) -> List[StrategyConfig]:
        """Get all configurations for a strategy or all strategies"""
        try:
            configs = []
            
            if strategy_id:
                # Get configs for specific strategy
                db_configs = await self.db.get_strategy_config(strategy_id)
                if db_configs:
                    configs.append(StrategyConfig.from_dict(db_configs))
            else:
                # Get all configs from local cache
                configs = list(self.configs.values())
            
            return configs
            
        except Exception as e:
            self.logger.error(f"Error getting all configs: {e}")
            return []
    
    def validate_signal(self, signal: Dict[str, Any], config: StrategyConfig) -> Dict[str, Any]:
        """Validate signal against configuration parameters"""
        try:
            validation_result = {
                'valid': True,
                'warnings': [],
                'errors': [],
                'adjusted_params': {}
            }
            
            # Check confidence threshold
            if signal.get('confidence', 0) < config.confidence_threshold:
                validation_result['warnings'].append(
                    f"Signal confidence {signal.get('confidence', 0):.3f} below threshold {config.confidence_threshold}"
                )
            
            # Check correlation threshold
            if 'correlation' in signal and abs(signal['correlation']) < config.correlation_threshold:
                validation_result['warnings'].append(
                    f"Correlation {signal.get('correlation', 0):.3f} below threshold {config.correlation_threshold}"
                )
            
            # Check volatility threshold
            if 'volatility' in signal and signal['volatility'] < config.volatility_threshold:
                validation_result['warnings'].append(
                    f"Volatility {signal.get('volatility', 0):.3f} below threshold {config.volatility_threshold}"
                )
            
            # Check funding rate threshold
            if 'funding_rate' in signal and abs(signal['funding_rate']) < config.funding_rate_threshold:
                validation_result['warnings'].append(
                    f"Funding rate {signal.get('funding_rate', 0):.6f} below threshold {config.funding_rate_threshold}"
                )
            
            # Adjust position size if needed
            if 'position_size' in signal:
                adjusted_size = min(signal['position_size'], config.position_size_pct)
                if adjusted_size != signal['position_size']:
                    validation_result['adjusted_params']['position_size'] = adjusted_size
                    validation_result['warnings'].append(
                        f"Position size adjusted from {signal['position_size']:.3f} to {adjusted_size:.3f}"
                    )
            
            # Check if warnings should be treated as errors
            if len(validation_result['warnings']) > 3:
                validation_result['valid'] = False
                validation_result['errors'].extend(validation_result['warnings'][:3])
                validation_result['warnings'] = validation_result['warnings'][3:]
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return {
                'valid': False,
                'warnings': [],
                'errors': [f"Validation error: {e}"],
                'adjusted_params': {}
            }
    
    async def close(self):
        """Close the configuration manager"""
        try:
            self.configs.clear()
            self.logger.info("Strategy configuration manager closed")
        except Exception as e:
            self.logger.error(f"Error closing config manager: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
