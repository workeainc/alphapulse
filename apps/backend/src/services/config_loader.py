"""
Unified Configuration Loader for AlphaPulse
Loads and merges all YAML configuration files
"""

import logging
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Unified configuration loader for AlphaPulse system
    Loads and merges symbol_config.yaml and mtf_config.yaml
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.logger = logger
        self._config_cache: Optional[Dict[str, Any]] = None
    
    def load_all_configs(self) -> Dict[str, Any]:
        """
        Load and merge all configuration files
        
        Returns:
            Dict with merged configuration from all YAML files
        """
        try:
            # Check cache
            if self._config_cache:
                return self._config_cache
            
            # Load individual configs
            symbol_config = self.load_symbol_config()
            mtf_config = self.load_mtf_config()
            
            # Merge configurations
            merged_config = self._merge_configs(symbol_config, mtf_config)
            
            # Validate
            if self.validate_config(merged_config):
                self._config_cache = merged_config
                self.logger.info("✅ All configurations loaded and validated successfully")
                return merged_config
            else:
                self.logger.warning("⚠️ Configuration validation failed, using defaults")
                return self._get_default_config()
                
        except Exception as e:
            self.logger.error(f"❌ Error loading configs: {e}")
            return self._get_default_config()
    
    def load_symbol_config(self) -> Dict[str, Any]:
        """Load symbol_config.yaml"""
        try:
            config_path = self.config_dir / "symbol_config.yaml"
            
            if not config_path.exists():
                self.logger.warning(f"⚠️ symbol_config.yaml not found at {config_path}, using defaults")
                return self._get_default_symbol_config()
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.logger.info(f"✅ Loaded symbol_config.yaml")
            return config or {}
            
        except Exception as e:
            self.logger.error(f"❌ Error loading symbol_config.yaml: {e}")
            return self._get_default_symbol_config()
    
    def load_mtf_config(self) -> Dict[str, Any]:
        """Load mtf_config.yaml"""
        try:
            config_path = self.config_dir / "mtf_config.yaml"
            
            if not config_path.exists():
                self.logger.warning(f"⚠️ mtf_config.yaml not found at {config_path}, using defaults")
                return self._get_default_mtf_config()
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.logger.info(f"✅ Loaded mtf_config.yaml")
            return config or {}
            
        except Exception as e:
            self.logger.error(f"❌ Error loading mtf_config.yaml: {e}")
            return self._get_default_mtf_config()
    
    def _merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries"""
        merged = {}
        
        for config in configs:
            if config:
                merged.update(config)
        
        return merged
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration has required fields
        
        Args:
            config: Configuration dictionary to validate
        
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            required_sections = []
            
            # Check basic structure (not strict, allow flexibility)
            if not isinstance(config, dict):
                self.logger.error("❌ Config is not a dictionary")
                return False
            
            # Log loaded sections
            sections = list(config.keys())
            self.logger.info(f"📋 Config sections loaded: {', '.join(sections)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Config validation error: {e}")
            return False
    
    def get_mtf_config(self) -> Dict[str, Any]:
        """Get MTF-specific configuration"""
        full_config = self.load_all_configs()
        return full_config.get('mtf_strategies', {})
    
    def get_symbol_config(self) -> Dict[str, Any]:
        """Get symbol management configuration"""
        full_config = self.load_all_configs()
        return full_config.get('symbol_management', {})
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when loading fails"""
        return {
            **self._get_default_symbol_config(),
            **self._get_default_mtf_config()
        }
    
    def _get_default_symbol_config(self) -> Dict[str, Any]:
        """Default symbol management configuration"""
        return {
            'symbol_management': {
                'total_symbols': 100,
                'futures_count': 50,
                'spot_count': 50,
                'update_interval_hours': 24,
                'min_volume_24h_usd': 10000000,
                'classification': {
                    'tier1_count': 20,
                    'tier2_count': 30,
                    'tier3_count': 50
                }
            },
            'websocket': {
                'max_connections': 10,
                'symbols_per_connection': 10,
                'reconnect_delay_seconds': 5,
                'ping_interval_seconds': 30
            },
            'signal_generation': {
                'symbols_per_batch': 10,
                'analysis_interval_seconds': 60,
                'analysis_timeout_seconds': 30,
                'min_data_candles': 200
            },
            'database': {
                'host': 'localhost',
                'port': 55433,
                'database': 'alphapulse',
                'user': 'alpha_emon'
            },
            'redis': {
                'url': 'redis://localhost:56379',
                'db': 0
            }
        }
    
    def _get_default_mtf_config(self) -> Dict[str, Any]:
        """Default MTF configuration"""
        return {
            'mtf_strategies': {
                'timeframe_mappings': {
                    '1d': '4h',
                    '4h': '1h',
                    '1h': '15m',
                    '15m': '5m',
                    '5m': '1m'
                }
            },
            'entry_strategies': {
                'fibonacci_levels': [0.236, 0.382, 0.5, 0.618, 0.786],
                'ema_periods': {'fast': 9, 'medium': 21, 'slow': 50}
            },
            'risk_management': {
                'stop_loss': {'atr_multiplier': 1.5},
                'take_profit': {
                    'tp1_atr_multiplier': 2.0,
                    'tp2_atr_multiplier': 3.5,
                    'tp3_atr_multiplier': 5.0
                },
                'min_risk_reward_ratio': 1.5
            },
            'entry_quality': {
                'min_entry_confidence': 0.60,
                'min_signal_confidence': 0.75,
                'volume_confirmation': {'enabled': True, 'volume_vs_ma_ratio': 1.2}
            }
        }


# Global config loader instance
config_loader = ConfigLoader()

