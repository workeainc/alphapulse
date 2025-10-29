"""
Data Quality Validator for Real Market Data
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates real market data quality"""
    
    def __init__(self):
        self.max_price_change = 0.5  # 50% max price change
        self.min_volume = 0.0
        self.max_timestamp_drift = 60  # 60 seconds
        self.price_history = {}
        
    def validate_market_data(self, data: Dict[str, Any]) -> bool:
        """Validate market data quality"""
        try:
            # Check required fields
            required_fields = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
            for field in required_fields:
                if field not in data:
                    logger.error(f"Missing required field: {field}")
                    return False
            
            # Validate price data
            if not self._validate_price_data(data):
                return False
            
            # Validate volume data
            if not self._validate_volume_data(data):
                return False
            
            # Validate timestamp
            if not self._validate_timestamp(data):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return False
    
    def _validate_price_data(self, data: Dict[str, Any]) -> bool:
        """Validate price data"""
        try:
            open_price = float(data['open'])
            high_price = float(data['high'])
            low_price = float(data['low'])
            close_price = float(data['close'])
            
            # Check for valid prices
            if any(price <= 0 for price in [open_price, high_price, low_price, close_price]):
                logger.error("Invalid price data: prices must be positive")
                return False
            
            # Check OHLC relationships
            if high_price < max(open_price, close_price):
                logger.error("Invalid OHLC: high < max(open, close)")
                return False
            
            if low_price > min(open_price, close_price):
                logger.error("Invalid OHLC: low > min(open, close)")
                return False
            
            # Check for extreme price changes
            symbol = data['symbol']
            if symbol in self.price_history:
                last_price = self.price_history[symbol]
                price_change = abs(close_price - last_price) / last_price
                
                if price_change > self.max_price_change:
                    logger.warning(f"Extreme price change detected: {price_change:.2%}")
                    return False
            
            # Update price history
            self.price_history[symbol] = close_price
            
            return True
            
        except Exception as e:
            logger.error(f"Price validation error: {e}")
            return False
    
    def _validate_volume_data(self, data: Dict[str, Any]) -> bool:
        """Validate volume data"""
        try:
            volume = float(data['volume'])
            
            if volume < self.min_volume:
                logger.error(f"Invalid volume: {volume}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Volume validation error: {e}")
            return False
    
    def _validate_timestamp(self, data: Dict[str, Any]) -> bool:
        """Validate timestamp"""
        try:
            timestamp = data['timestamp']
            current_time = datetime.utcnow()
            
            # Check timestamp drift
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            time_diff = abs((current_time - timestamp).total_seconds())
            
            if time_diff > self.max_timestamp_drift:
                logger.warning(f"Timestamp drift detected: {time_diff} seconds")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Timestamp validation error: {e}")
            return False
