#!/usr/bin/env python3
"""
Environment setup script for AlphaPlus Trading System.
This script sets up the environment configuration for the trading system.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

def setup_environment():
    """Setup environment configuration"""
    # Update path to new config location
    template_file = 'config/config/config/config.env.template'
    
    if not os.path.exists(template_file):
        print(f"❌ Template file not found: {template_file}")
        print("Please ensure the config directory exists and contains config/config/config.env.template")
        return False
    
    # Create config directory if it doesn't exist
    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)
    
    # Copy template to actual config file
    config_file = 'config/config.env'
    if not os.path.exists(config_file):
        shutil.copy2(template_file, config_file)
        print(f"✅ Created config file: {config_file}")
    else:
        print(f"ℹ️  Config file already exists: {config_file}")
    
    # Set up environment variables
    env_vars = {
        'ALPHAPULSE_ENV': 'development',
        'LOG_LEVEL': 'INFO',
        'DATABASE_URL': 'postgresql://user:password@localhost:5432/alphapulse',
        'REDIS_URL': 'redis://localhost:6379/0',
        'BINANCE_API_KEY': 'your_api_key_here',
        'BINANCE_SECRET_KEY': 'your_secret_key_here',
        'TRADING_ENABLED': 'false',
        'RISK_MANAGEMENT_ENABLED': 'true',
        'MAX_POSITION_SIZE': '0.1',
        'MAX_DAILY_LOSS': '0.05',
        'BACKTESTING_MODE': 'true'
    }
    
    # Update config file with environment variables
    update_config_file(config_file, env_vars)
    
    print("✅ Environment setup completed successfully!")
    print("📝 Please review and update the configuration values in config/config.env")
    
    return True

def update_config_file(config_file: str, env_vars: Dict[str, Any]) -> None:
    """Update configuration file with environment variables"""
    try:
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Update existing values or add new ones
        for key, value in env_vars.items():
            if f"{key}=" in content:
                # Update existing value
                content = content.replace(f"{key}=", f"{key}={value}")
            else:
                # Add new value
                content += f"\n{key}={value}"
        
        with open(config_file, 'w') as f:
            f.write(content)
            
    except Exception as e:
        print(f"⚠️  Warning: Could not update config file: {e}")

def validate_environment() -> bool:
    """Validate that the environment is properly configured"""
    config_file = 'config/config.env'
    
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        return False
    
    # Check for required environment variables
    required_vars = [
        'DATABASE_URL',
        'REDIS_URL',
        'BINANCE_API_KEY',
        'BINANCE_SECRET_KEY'
    ]
    
    missing_vars = []
    with open(config_file, 'r') as f:
        content = f.read()
        for var in required_vars:
            if f"{var}=" not in content or f"{var}=" in content and f"{var}=your_" in content:
                missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing or incomplete configuration for: {', '.join(missing_vars)}")
        return False
    
    print("✅ Environment validation passed!")
    return True

def main():
    """Main function"""
    print("🚀 Setting up AlphaPlus Trading System environment...")
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # Validate environment
    if not validate_environment():
        print("⚠️  Environment setup completed with warnings.")
        print("Please review and update the configuration values.")
    else:
        print("🎉 Environment setup completed successfully!")

if __name__ == "__main__":
    main()

