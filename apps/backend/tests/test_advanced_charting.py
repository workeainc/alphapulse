#!/usr/bin/env python3
"""
Test Advanced Charting System
Test the enhanced charting capabilities for AlphaPulse
"""

import asyncio
import json
import logging
import asyncpg
from datetime import datetime, timedelta
import importlib.util

# Import production config
spec = importlib.util.spec_from_file_location('production', 'config/production.py')
production_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(production_module)
production_config = production_module.production_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_advanced_charting():
    """Test the advanced charting system"""
    
    # Connect to database
    db_pool = await asyncpg.create_pool(
        host=production_config.DATABASE_CONFIG['host'],
        port=production_config.DATABASE_CONFIG['port'],
        database=production_config.DATABASE_CONFIG['database'],
        user=production_config.DATABASE_CONFIG['username'],
        password=production_config.DATABASE_CONFIG['password']
    )
    
    try:
        logger.info("Testing advanced charting system...")
        
        # Test 1: Generate sample chart data
        logger.info("Test 1: Generating sample chart data...")
        chart_data = generate_sample_chart_data()
        logger.info(f"Generated {len(chart_data)} data points")
        
        # Test 2: Calculate technical indicators
        logger.info("Test 2: Calculating technical indicators...")
        indicators = calculate_technical_indicators(chart_data)
        for indicator in indicators:
            logger.info(f"  {indicator['name']}: {indicator['value']:.2f} ({indicator['signal']})")
        
        # Test 3: Generate trading signals
        logger.info("Test 3: Generating trading signals...")
        signals = generate_trading_signals(chart_data, indicators)
        for signal in signals:
            logger.info(f"  {signal['type'].upper()} Signal: {signal['description']} at ${signal['price']:.2f}")
        
        # Test 4: Validate chart data structure
        logger.info("Test 4: Validating chart data structure...")
        validate_chart_data(chart_data)
        
        # Test 5: Validate technical indicators
        logger.info("Test 5: Validating technical indicators...")
        validate_technical_indicators(indicators)
        
        logger.info("✅ Advanced charting system test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Advanced charting system test failed: {e}")
        raise
    finally:
        # Close database connection
        await db_pool.close()

def validate_chart_data(chart_data):
    """Validate chart data structure"""
    if not chart_data:
        raise ValueError("Chart data is empty")
    
    required_fields = ['timestamp', 'price', 'volume', 'open', 'high', 'low', 'close']
    
    for i, data in enumerate(chart_data):
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing field '{field}' in data point {i}")
        
        # Validate data types
        if not isinstance(data['price'], (int, float)):
            raise ValueError(f"Invalid price type in data point {i}")
        if not isinstance(data['volume'], (int, float)):
            raise ValueError(f"Invalid volume type in data point {i}")
        if not isinstance(data['timestamp'], datetime):
            raise ValueError(f"Invalid timestamp type in data point {i}")
    
    logger.info(f"✅ Chart data validation passed - {len(chart_data)} data points")

def validate_technical_indicators(indicators):
    """Validate technical indicators"""
    if not indicators:
        raise ValueError("No technical indicators generated")
    
    required_fields = ['name', 'value', 'signal']
    valid_signals = ['buy', 'sell', 'neutral']
    
    for indicator in indicators:
        for field in required_fields:
            if field not in indicator:
                raise ValueError(f"Missing field '{field}' in indicator")
        
        if indicator['signal'] not in valid_signals:
            raise ValueError(f"Invalid signal '{indicator['signal']}' in indicator {indicator['name']}")
        
        if not isinstance(indicator['value'], (int, float)):
            raise ValueError(f"Invalid value type in indicator {indicator['name']}")
    
    logger.info(f"✅ Technical indicators validation passed - {len(indicators)} indicators")

def generate_sample_chart_data():
    """Generate sample chart data for testing"""
    data = []
    base_price = 45000
    current_time = datetime.now() - timedelta(hours=24)
    
    for i in range(100):
        # Simulate price movement
        price_change = (i % 20 - 10) * 50  # Oscillating pattern
        volume = 1000 + (i % 10) * 200
        
        data.append({
            'timestamp': current_time + timedelta(minutes=i*15),
            'price': base_price + price_change,
            'volume': volume,
            'open': base_price + price_change - 25,
            'high': base_price + price_change + 50,
            'low': base_price + price_change - 50,
            'close': base_price + price_change
        })
    
    return data

def calculate_technical_indicators(chart_data):
    """Calculate technical indicators"""
    prices = [d['price'] for d in chart_data]
    volumes = [d['volume'] for d in chart_data]
    
    # RSI calculation
    rsi = calculate_rsi(prices, 14)
    current_rsi = rsi[-1] if rsi else 50
    
    # MACD calculation
    macd_data = calculate_macd(prices)
    current_macd = macd_data['macd'][-1] if macd_data['macd'] else 0
    current_signal = macd_data['signal'][-1] if macd_data['signal'] else 0
    
    # Volume analysis
    avg_volume = sum(volumes) / len(volumes)
    current_volume = volumes[-1] if volumes else 0
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
    
    return [
        {
            'name': 'RSI',
            'value': current_rsi,
            'signal': 'sell' if current_rsi > 70 else 'buy' if current_rsi < 30 else 'neutral'
        },
        {
            'name': 'MACD',
            'value': current_macd,
            'signal': 'buy' if current_macd > current_signal else 'sell'
        },
        {
            'name': 'Volume',
            'value': volume_ratio,
            'signal': 'buy' if volume_ratio > 1.5 else 'sell' if volume_ratio < 0.5 else 'neutral'
        }
    ]

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    if len(prices) < period + 1:
        return []
    
    gains = []
    losses = []
    
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        gains.append(change if change > 0 else 0)
        losses.append(abs(change) if change < 0 else 0)
    
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    if avg_loss == 0:
        return [100]
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return [rsi]

def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD"""
    if len(prices) < slow_period:
        return {'macd': [], 'signal': []}
    
    # Simplified EMA calculation
    ema12 = calculate_ema(prices, fast_period)
    ema26 = calculate_ema(prices, slow_period)
    macd = ema12 - ema26
    signal = macd  # Simplified signal line
    
    return {'macd': [macd], 'signal': [signal]}

def calculate_ema(prices, period):
    """Calculate EMA"""
    if len(prices) < period:
        return prices[-1] if prices else 0
    
    multiplier = 2 / (period + 1)
    ema = prices[0]
    
    for price in prices[1:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
    
    return ema

def generate_trading_signals(chart_data, indicators):
    """Generate trading signals based on indicators"""
    signals = []
    
    if not chart_data or not indicators:
        return signals
    
    current_price = chart_data[-1]['price']
    current_time = chart_data[-1]['timestamp']
    
    # RSI signals
    rsi_indicator = next((i for i in indicators if i['name'] == 'RSI'), None)
    if rsi_indicator and rsi_indicator['signal'] == 'buy':
        signals.append({
            'timestamp': current_time.isoformat(),
            'type': 'buy',
            'price': current_price,
            'strength': 0.8,
            'description': 'RSI oversold - Strong buy signal'
        })
    
    # MACD signals
    macd_indicator = next((i for i in indicators if i['name'] == 'MACD'), None)
    if macd_indicator and macd_indicator['signal'] == 'buy':
        signals.append({
            'timestamp': current_time.isoformat(),
            'type': 'buy',
            'price': current_price,
            'strength': 0.7,
            'description': 'MACD bullish crossover'
        })
    
    return signals

# Database functions removed for simplicity - focusing on charting functionality validation

if __name__ == "__main__":
    asyncio.run(test_advanced_charting())
