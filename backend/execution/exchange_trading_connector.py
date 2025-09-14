#!/usr/bin/env python3
"""
Exchange Trading Connector for AlphaPulse
Handles actual order execution on real exchanges
"""

import asyncio
import aiohttp
import hmac
import hashlib
import time
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import urllib.parse

logger = logging.getLogger(__name__)

class ExchangeType(Enum):
    """Supported exchange types"""
    BINANCE = "binance"
    BYBIT = "bybit"
    COINBASE = "coinbase"
    KUCOIN = "kucoin"

class OrderExecutionStatus(Enum):
    """Order execution status"""
    PENDING = "pending"
    EXECUTING = "executing"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ExchangeCredentials:
    """Exchange API credentials"""
    api_key: str
    secret_key: str
    passphrase: Optional[str] = None  # For Coinbase
    testnet: bool = False

@dataclass
class ExecutionResult:
    """Result of order execution"""
    success: bool
    order_id: Optional[str] = None
    exchange_order_id: Optional[str] = None
    executed_quantity: float = 0.0
    executed_price: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0
    execution_time_ms: int = 0
    error: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class BaseExchangeTradingConnector:
    """Base class for exchange trading connectors"""
    
    def __init__(self, exchange_type: ExchangeType, credentials: ExchangeCredentials):
        self.exchange_type = exchange_type
        self.credentials = credentials
        self.session = None
        self.base_url = ""
        self.testnet_url = ""
        
        # Rate limiting
        self.rate_limit_requests = 0
        self.rate_limit_window = 0
        self.max_requests_per_window = 100
        
        # Performance tracking
        self.total_orders = 0
        self.successful_orders = 0
        self.failed_orders = 0
        self.avg_execution_time = 0.0
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def place_market_order(self, symbol: str, side: str, quantity: float) -> ExecutionResult:
        """Place a market order"""
        raise NotImplementedError("Subclasses must implement place_market_order")
    
    async def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> ExecutionResult:
        """Place a limit order"""
        raise NotImplementedError("Subclasses must implement place_limit_order")
    
    async def place_stop_order(self, symbol: str, side: str, quantity: float, stop_price: float) -> ExecutionResult:
        """Place a stop order"""
        raise NotImplementedError("Subclasses must implement place_stop_order")
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an existing order"""
        raise NotImplementedError("Subclasses must implement cancel_order")
    
    async def get_order_status(self, symbol: str, order_id: str) -> Dict:
        """Get order status"""
        raise NotImplementedError("Subclasses must implement get_order_status")
    
    async def get_account_balance(self) -> Dict[str, float]:
        """Get account balance"""
        raise NotImplementedError("Subclasses must implement get_account_balance")
    
    def _sign_request(self, params: str, secret: str) -> str:
        """Sign request with HMAC SHA256"""
        return hmac.new(
            secret.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        current_time = time.time()
        if current_time - self.rate_limit_window > 60:  # 1 minute window
            self.rate_limit_requests = 0
            self.rate_limit_window = current_time
        
        if self.rate_limit_requests >= self.max_requests_per_window:
            return False
        
        self.rate_limit_requests += 1
        return True
    
    async def _wait_for_rate_limit(self):
        """Wait if rate limit exceeded"""
        while not self._check_rate_limit():
            await asyncio.sleep(1)
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            "total_orders": self.total_orders,
            "successful_orders": self.successful_orders,
            "failed_orders": self.failed_orders,
            "success_rate": self.successful_orders / max(self.total_orders, 1),
            "avg_execution_time": self.avg_execution_time
        }

class BinanceTradingConnector(BaseExchangeTradingConnector):
    """Binance trading connector"""
    
    def __init__(self, credentials: ExchangeCredentials):
        super().__init__(ExchangeType.BINANCE, credentials)
        if credentials.testnet:
            self.base_url = "https://testnet.binance.vision"
        else:
            self.base_url = "https://api.binance.com"
    
    async def place_market_order(self, symbol: str, side: str, quantity: float) -> ExecutionResult:
        """Place a market order on Binance"""
        start_time = time.time()
        
        try:
            await self._wait_for_rate_limit()
            
            # Prepare parameters
            params = {
                "symbol": symbol,
                "side": side.upper(),
                "type": "MARKET",
                "quantity": quantity,
                "timestamp": int(time.time() * 1000)
            }
            
            # Sign request
            query_string = urllib.parse.urlencode(params)
            signature = self._sign_request(query_string, self.credentials.secret_key)
            params["signature"] = signature
            
            # Make request
            headers = {"X-MBX-APIKEY": self.credentials.api_key}
            url = f"{self.base_url}/api/v3/order"
            
            async with self.session.post(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    execution_time = int((time.time() - start_time) * 1000)
                    
                    result = ExecutionResult(
                        success=True,
                        order_id=data.get("orderId"),
                        exchange_order_id=data.get("orderId"),
                        executed_quantity=float(data.get("executedQty", 0)),
                        executed_price=float(data.get("avgPrice", 0)) if data.get("avgPrice") else None,
                        commission=float(data.get("commission", 0)),
                        execution_time_ms=execution_time
                    )
                    
                    self.total_orders += 1
                    self.successful_orders += 1
                    self.avg_execution_time = (
                        (self.avg_execution_time * (self.successful_orders - 1) + execution_time) / 
                        self.successful_orders
                    )
                    
                    return result
                else:
                    error_data = await response.json()
                    self.total_orders += 1
                    self.failed_orders += 1
                    
                    return ExecutionResult(
                        success=False,
                        error=f"HTTP {response.status}: {error_data.get('msg', 'Unknown error')}"
                    )
                    
        except Exception as e:
            self.total_orders += 1
            self.failed_orders += 1
            logger.error(f"Error placing market order: {e}")
            
            return ExecutionResult(
                success=False,
                error=str(e)
            )
    
    async def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> ExecutionResult:
        """Place a limit order on Binance"""
        start_time = time.time()
        
        try:
            await self._wait_for_rate_limit()
            
            # Prepare parameters
            params = {
                "symbol": symbol,
                "side": side.upper(),
                "type": "LIMIT",
                "timeInForce": "GTC",
                "quantity": quantity,
                "price": price,
                "timestamp": int(time.time() * 1000)
            }
            
            # Sign request
            query_string = urllib.parse.urlencode(params)
            signature = self._sign_request(query_string, self.credentials.secret_key)
            params["signature"] = signature
            
            # Make request
            headers = {"X-MBX-APIKEY": self.credentials.api_key}
            url = f"{self.base_url}/api/v3/order"
            
            async with self.session.post(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    execution_time = int((time.time() - start_time) * 1000)
                    
                    result = ExecutionResult(
                        success=True,
                        order_id=data.get("orderId"),
                        exchange_order_id=data.get("orderId"),
                        executed_quantity=0.0,  # Limit orders start unfilled
                        execution_time_ms=execution_time
                    )
                    
                    self.total_orders += 1
                    self.successful_orders += 1
                    self.avg_execution_time = (
                        (self.avg_execution_time * (self.successful_orders - 1) + execution_time) / 
                        self.successful_orders
                    )
                    
                    return result
                else:
                    error_data = await response.json()
                    self.total_orders += 1
                    self.failed_orders += 1
                    
                    return ExecutionResult(
                        success=False,
                        error=f"HTTP {response.status}: {error_data.get('msg', 'Unknown error')}"
                    )
                    
        except Exception as e:
            self.total_orders += 1
            self.failed_orders += 1
            logger.error(f"Error placing limit order: {e}")
            
            return ExecutionResult(
                success=False,
                error=str(e)
            )
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order on Binance"""
        try:
            await self._wait_for_rate_limit()
            
            params = {
                "symbol": symbol,
                "orderId": order_id,
                "timestamp": int(time.time() * 1000)
            }
            
            # Sign request
            query_string = urllib.parse.urlencode(params)
            signature = self._sign_request(query_string, self.credentials.secret_key)
            params["signature"] = signature
            
            headers = {"X-MBX-APIKEY": self.credentials.api_key}
            url = f"{self.base_url}/api/v3/order"
            
            async with self.session.delete(url, params=params, headers=headers) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    async def get_order_status(self, symbol: str, order_id: str) -> Dict:
        """Get order status from Binance"""
        try:
            await self._wait_for_rate_limit()
            
            params = {
                "symbol": symbol,
                "orderId": order_id,
                "timestamp": int(time.time() * 1000)
            }
            
            # Sign request
            query_string = urllib.parse.urlencode(params)
            signature = self._sign_request(query_string, self.credentials.secret_key)
            params["signature"] = signature
            
            headers = {"X-MBX-APIKEY": self.credentials.api_key}
            url = f"{self.base_url}/api/v3/order"
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}"}
                    
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return {"error": str(e)}
    
    async def get_account_balance(self) -> Dict[str, float]:
        """Get account balance from Binance"""
        try:
            await self._wait_for_rate_limit()
            
            params = {
                "timestamp": int(time.time() * 1000)
            }
            
            # Sign request
            query_string = urllib.parse.urlencode(params)
            signature = self._sign_request(query_string, self.credentials.secret_key)
            params["signature"] = signature
            
            headers = {"X-MBX-APIKEY": self.credentials.api_key}
            url = f"{self.base_url}/api/v3/account"
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    balances = {}
                    
                    for balance in data.get("balances", []):
                        asset = balance["asset"]
                        free = float(balance["free"])
                        locked = float(balance["locked"])
                        total = free + locked
                        
                        if total > 0:
                            balances[asset] = total
                    
                    return balances
                else:
                    return {"error": f"HTTP {response.status}"}
                    
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return {"error": str(e)}

class BybitTradingConnector(BaseExchangeTradingConnector):
    """Bybit trading connector"""
    
    def __init__(self, credentials: ExchangeCredentials):
        super().__init__(ExchangeType.BYBIT, credentials)
        if credentials.testnet:
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"
    
    async def place_market_order(self, symbol: str, side: str, quantity: float) -> ExecutionResult:
        """Place a market order on Bybit"""
        # Implementation similar to Binance but with Bybit API specifics
        # This is a placeholder - full implementation would follow Bybit's API structure
        raise NotImplementedError("Bybit implementation pending")
    
    async def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> ExecutionResult:
        """Place a limit order on Bybit"""
        raise NotImplementedError("Bybit implementation pending")
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order on Bybit"""
        raise NotImplementedError("Bybit implementation pending")
    
    async def get_order_status(self, symbol: str, order_id: str) -> Dict:
        """Get order status from Bybit"""
        raise NotImplementedError("Bybit implementation pending")
    
    async def get_account_balance(self) -> Dict[str, float]:
        """Get account balance from Bybit"""
        raise NotImplementedError("Bybit implementation pending")

def create_trading_connector(exchange_type: ExchangeType, credentials: ExchangeCredentials) -> BaseExchangeTradingConnector:
    """Factory function to create trading connectors"""
    if exchange_type == ExchangeType.BINANCE:
        return BinanceTradingConnector(credentials)
    elif exchange_type == ExchangeType.BYBIT:
        return BybitTradingConnector(credentials)
    else:
        raise ValueError(f"Unsupported exchange type: {exchange_type}")

async def test_binance_connector():
    """Test the Binance trading connector"""
    # Note: This requires real API credentials
    credentials = ExchangeCredentials(
        api_key="your_api_key_here",
        secret_key="your_secret_key_here",
        testnet=True  # Use testnet for testing
    )
    
    async with create_trading_connector(ExchangeType.BINANCE, credentials) as connector:
        # Test account balance (read-only)
        balance = await connector.get_account_balance()
        print(f"Account balance: {balance}")
        
        # Test performance stats
        stats = connector.get_performance_stats()
        print(f"Performance stats: {stats}")

class ExchangeTradingConnector:
    """Concrete exchange trading connector for AlphaPlus"""
    
    def __init__(self):
        self.logger = logger
        self.connected = False
        self.test_mode = True  # Start in test mode
    
    async def initialize(self):
        """Initialize the connector"""
        try:
            self.logger.info("Initializing Exchange Trading Connector...")
            self.connected = True
            self.logger.info("Exchange Trading Connector initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize connector: {e}")
            raise
    
    async def execute_order(self, order) -> Dict[str, Any]:
        """Execute an order (simulated for now)"""
        try:
            # Simulate order execution
            await asyncio.sleep(0.1)  # Simulate network delay
            
            return {
                'status': 'success',
                'filled_quantity': order.quantity,
                'fill_price': order.price or 100.0,  # Default price for testing
                'commission': 0.001,  # 0.1% commission
                'position_id': f"pos_{order.id}"
            }
        except Exception as e:
            self.logger.error(f"Error executing order: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order (simulated for now)"""
        try:
            await asyncio.sleep(0.05)  # Simulate network delay
            
            return {
                'status': 'success',
                'order_id': order_id
            }
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol (simulated for now)"""
        try:
            # Simulate price data
            import random
            base_price = 100.0 if 'BTC' in symbol else 50.0
            return base_price + random.uniform(-5, 5)
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the connector"""
        try:
            return {
                'status': 'healthy' if self.connected else 'disconnected',
                'connected': self.connected,
                'test_mode': self.test_mode
            }
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            return {'status': 'error', 'error': str(e)}


if __name__ == "__main__":
    asyncio.run(test_binance_connector())
