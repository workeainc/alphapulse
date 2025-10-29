#!/usr/bin/env python3
"""
Safe Exchange Configuration
Provides consistent exchange configuration across the application
"""

import ccxt
import logging

logger = logging.getLogger(__name__)

def create_safe_binance_exchange() -> ccxt.binance:
    """
    Create a safely configured Binance exchange instance
    Ensures spot API is used consistently
    """
    try:
        exchange = ccxt.binance({
            'sandbox': False,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # Force spot API
                'adjustForTimeDifference': True,
                'recvWindow': 60000,
                'defaultTimeInForce': 'GTC'
            },
            'timeout': 30000,
            'rateLimit': 1200,  # 1200 requests per minute
        })
        
        # Force spot API endpoints and disable futures
        exchange.urls['api'] = {
            'public': 'https://api.binance.com/api/v3',
            'private': 'https://api.binance.com/api/v3',
            'v1': 'https://api.binance.com/api/v1',
            'v3': 'https://api.binance.com/api/v3',
        }
        
        # Remove futures endpoints completely
        if 'fapiPublic' in exchange.urls['api']:
            del exchange.urls['api']['fapiPublic']
        if 'fapiPrivate' in exchange.urls['api']:
            del exchange.urls['api']['fapiPrivate']
        if 'dapiPublic' in exchange.urls['api']:
            del exchange.urls['api']['dapiPublic']
        if 'dapiPrivate' in exchange.urls['api']:
            del exchange.urls['api']['dapiPrivate']
        
        logger.info("✅ Binance exchange configured for SPOT API")
        return exchange
        
    except Exception as e:
        logger.error(f"❌ Error creating Binance exchange: {e}")
        raise

def create_safe_bybit_exchange() -> ccxt.bybit:
    """
    Create a safely configured Bybit exchange instance
    """
    try:
        exchange = ccxt.bybit({
            'sandbox': False,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True
            },
            'timeout': 30000,
            'rateLimit': 1200,
        })
        
        logger.info("✅ Bybit exchange configured for SPOT API")
        return exchange
        
    except Exception as e:
        logger.error(f"❌ Error creating Bybit exchange: {e}")
        raise

def create_safe_okx_exchange() -> ccxt.okx:
    """
    Create a safely configured OKX exchange instance
    """
    try:
        exchange = ccxt.okx({
            'sandbox': False,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True
            },
            'timeout': 30000,
            'rateLimit': 1200,
        })
        
        logger.info("✅ OKX exchange configured for SPOT API")
        return exchange
        
    except Exception as e:
        logger.error(f"❌ Error creating OKX exchange: {e}")
        raise

def test_exchange_connection(exchange: ccxt.Exchange, symbol: str = 'BTC/USDT') -> bool:
    """
    Test if exchange connection is working
    """
    try:
        # Test with a simple API call
        ticker = exchange.fetch_ticker(symbol)
        logger.info(f"✅ Exchange test successful: {ticker['symbol']} = ${ticker['last']}")
        return True
    except Exception as e:
        logger.error(f"❌ Exchange test failed: {e}")
        return False

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Test Binance
        print("Testing Binance...")
        binance = create_safe_binance_exchange()
        success = test_exchange_connection(binance)
        print(f"Binance test: {'✅ PASS' if success else '❌ FAIL'}")
        
        # Test Bybit
        print("\nTesting Bybit...")
        bybit = create_safe_bybit_exchange()
        success = test_exchange_connection(bybit)
        print(f"Bybit test: {'✅ PASS' if success else '❌ FAIL'}")
        
        # Test OKX
        print("\nTesting OKX...")
        okx = create_safe_okx_exchange()
        success = test_exchange_connection(okx)
        print(f"OKX test: {'✅ PASS' if success else '❌ FAIL'}")
    
    asyncio.run(main())
