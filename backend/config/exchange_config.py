#!/usr/bin/env python3
"""
Exchange Configuration for AlphaPulse Trading System
Contains API credentials and exchange-specific settings
"""

import os
from typing import Dict, Optional
from dataclasses import dataclass
from execution.exchange_trading_connector import ExchangeType, ExchangeCredentials

@dataclass
class ExchangeConfig:
    """Configuration for a specific exchange"""
    name: str
    api_key: str
    secret_key: str
    passphrase: Optional[str] = None
    testnet: bool = True  # Default to testnet for safety
    base_url: str = ""
    testnet_url: str = ""
    
    def to_credentials(self) -> ExchangeCredentials:
        """Convert to ExchangeCredentials object"""
        return ExchangeCredentials(
            api_key=self.api_key,
            secret_key=self.secret_key,
            passphrase=self.passphrase,
            testnet=self.testnet
        )

class ExchangeConfigManager:
    """Manages exchange configurations"""
    
    def __init__(self):
        self.exchanges: Dict[str, ExchangeConfig] = {}
        self._load_configs()
    
    def _load_configs(self):
        """Load exchange configurations from environment variables"""
        # Binance configuration
        binance_api_key = os.getenv('BINANCE_API_KEY', '')
        binance_secret = os.getenv('BINANCE_SECRET_KEY', '')
        
        if binance_api_key and binance_secret:
            self.exchanges['binance'] = ExchangeConfig(
                name='binance',
                api_key=binance_api_key,
                secret_key=binance_secret,
                testnet=os.getenv('BINANCE_TESTNET', 'true').lower() == 'true',
                base_url='https://api.binance.com',
                testnet_url='https://testnet.binance.vision'
            )
        
        # Bybit configuration
        bybit_api_key = os.getenv('BYBIT_API_KEY', '')
        bybit_secret = os.getenv('BYBIT_SECRET_KEY', '')
        
        if bybit_api_key and bybit_secret:
            self.exchanges['bybit'] = ExchangeConfig(
                name='bybit',
                api_key=bybit_api_key,
                secret_key=bybit_secret,
                testnet=os.getenv('BYBIT_TESTNET', 'true').lower() == 'true',
                base_url='https://api.bybit.com',
                testnet_url='https://api-testnet.bybit.com'
            )
        
        # Coinbase configuration
        coinbase_api_key = os.getenv('COINBASE_API_KEY', '')
        coinbase_secret = os.getenv('COINBASE_SECRET_KEY', '')
        coinbase_passphrase = os.getenv('COINBASE_PASSPHRASE', '')
        
        if coinbase_api_key and coinbase_secret:
            self.exchanges['coinbase'] = ExchangeConfig(
                name='coinbase',
                api_key=coinbase_api_key,
                secret_key=coinbase_secret,
                passphrase=coinbase_passphrase,
                testnet=os.getenv('COINBASE_TESTNET', 'true').lower() == 'true',
                base_url='https://api.exchange.coinbase.com',
                testnet_url='https://api-public.sandbox.exchange.coinbase.com'
            )
    
    def get_exchange_config(self, exchange_name: str) -> Optional[ExchangeConfig]:
        """Get configuration for a specific exchange"""
        return self.exchanges.get(exchange_name.lower())
    
    def get_all_configs(self) -> Dict[str, ExchangeConfig]:
        """Get all exchange configurations"""
        return self.exchanges.copy()
    
    def is_exchange_configured(self, exchange_name: str) -> bool:
        """Check if an exchange is configured"""
        return exchange_name.lower() in self.exchanges
    
    def get_primary_exchange(self) -> Optional[ExchangeConfig]:
        """Get the primary exchange configuration (first available)"""
        if self.exchanges:
            return list(self.exchanges.values())[0]
        return None
    
    def add_exchange_config(self, config: ExchangeConfig):
        """Add a new exchange configuration"""
        self.exchanges[config.name.lower()] = config
    
    def remove_exchange_config(self, exchange_name: str):
        """Remove an exchange configuration"""
        if exchange_name.lower() in self.exchanges:
            del self.exchanges[exchange_name.lower()]
    
    def validate_configs(self) -> Dict[str, bool]:
        """Validate all exchange configurations"""
        validation_results = {}
        
        for name, config in self.exchanges.items():
            is_valid = (
                bool(config.api_key) and 
                bool(config.secret_key) and
                len(config.api_key) > 10 and
                len(config.secret_key) > 10
            )
            validation_results[name] = is_valid
        
        return validation_results
    
    def get_testnet_status(self) -> Dict[str, bool]:
        """Get testnet status for all exchanges"""
        return {name: config.testnet for name, config in self.exchanges.items()}

# Global instance
exchange_config_manager = ExchangeConfigManager()

def get_exchange_config(exchange_name: str) -> Optional[ExchangeConfig]:
    """Get exchange configuration by name"""
    return exchange_config_manager.get_exchange_config(exchange_name)

def get_primary_exchange_config() -> Optional[ExchangeConfig]:
    """Get primary exchange configuration"""
    return exchange_config_manager.get_primary_exchange_config()

def is_exchange_configured(exchange_name: str) -> bool:
    """Check if exchange is configured"""
    return exchange_config_manager.is_exchange_configured(exchange_name)

# Example usage and testing
if __name__ == "__main__":
    # Test configuration loading
    print("🔧 Exchange Configuration Status:")
    print("=" * 50)
    
    for name, config in exchange_config_manager.get_all_configs().items():
        status = "✅ Configured" if config.api_key and config.secret_key else "❌ Not Configured"
        testnet_status = "🧪 Testnet" if config.testnet else "🚀 Live"
        print(f"{name.upper()}: {status} ({testnet_status})")
    
    print("\n📋 Validation Results:")
    validation = exchange_config_manager.validate_configs()
    for exchange, is_valid in validation.items():
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"{exchange}: {status}")
    
    print("\n🧪 Testnet Status:")
    testnet_status = exchange_config_manager.get_testnet_status()
    for exchange, is_testnet in testnet_status.items():
        status = "🧪 Testnet" if is_testnet else "🚀 Live"
        print(f"{exchange}: {status}")
