import asyncio
import logging
import json
import re
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# from config import settings
# from ..src.database.connection import get_async_db
# from ..src.database.models import Trade

logger = logging.getLogger(__name__)

class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class SignalStrength(Enum):
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"

@dataclass
class PineScriptSignal:
    """Represents a signal from Pine Script"""
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    price: float
    timestamp: datetime
    strategy_name: str
    confidence: float
    metadata: Dict[str, Any]
    source: str = "pine_script"

class PineScriptProcessor:
    """Process and validate Pine Script signals"""
    
    def __init__(self):
        self.active_signals = {}
        self.signal_timeout = 300  # 5 minutes default
        self.update_interval = 60   # 1 minute default
        
    async def process_signal(self, signal_data: Dict) -> Optional[PineScriptSignal]:
        """Process incoming Pine Script signal"""
        try:
            # Validate signal format
            if not self._validate_signal_format(signal_data):
                logger.warning(f"Invalid signal format: {signal_data}")
                return None
            
            # Parse signal
            signal = self._parse_signal(signal_data)
            
            # Validate signal logic
            if not await self._validate_signal_logic(signal):
                logger.warning(f"Signal validation failed: {signal}")
                return None
            
            # Store signal
            self.active_signals[f"{signal.symbol}_{signal.strategy_name}"] = {
                "signal": signal,
                "timestamp": datetime.now()
            }
            
            logger.info(f"Processed Pine Script signal: {signal.symbol} {signal.signal_type.value} {signal.strength.value}")
            return signal
            
        except Exception as e:
            logger.error(f"Error processing Pine Script signal: {e}")
            return None
    
    def _validate_signal_format(self, signal_data: Dict) -> bool:
        """Validate signal data format"""
        required_fields = ["symbol", "signal_type", "price", "strategy_name"]
        
        for field in required_fields:
            if field not in signal_data:
                return False
        
        # Validate signal type
        if signal_data["signal_type"] not in [st.value for st in SignalType]:
            return False
        
        # Validate price
        try:
            float(signal_data["price"])
        except (ValueError, TypeError):
            return False
        
        return True
    
    def _parse_signal(self, signal_data: Dict) -> PineScriptSignal:
        """Parse signal data into PineScriptSignal object"""
        return PineScriptSignal(
            symbol=signal_data["symbol"],
            signal_type=SignalType(signal_data["signal_type"]),
            strength=SignalStrength(signal_data.get("strength", "medium")),
            price=float(signal_data["price"]),
            timestamp=datetime.fromisoformat(signal_data.get("timestamp", datetime.now().isoformat())),
            strategy_name=signal_data["strategy_name"],
            confidence=float(signal_data.get("confidence", 0.5)),
            metadata=signal_data.get("metadata", {}),
            source="pine_script"
        )
    
    async def _validate_signal_logic(self, signal: PineScriptSignal) -> bool:
        """Validate signal logic and market conditions"""
        try:
            # Check if signal is too old
            if datetime.now() - signal.timestamp > timedelta(seconds=self.signal_timeout):
                logger.warning(f"Signal too old: {signal.symbol}")
                return False
            
            # Check for duplicate recent signals
            if await self._is_duplicate_signal(signal):
                logger.info(f"Duplicate signal ignored: {signal.symbol}")
                return False
            
            # Validate price is reasonable
            if not await self._validate_price(signal):
                logger.warning(f"Price validation failed: {signal.symbol} at {signal.price}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal logic: {e}")
            return False
    
    async def _is_duplicate_signal(self, signal: PineScriptSignal) -> bool:
        """Check if this is a duplicate of a recent signal"""
        key = f"{signal.symbol}_{signal.strategy_name}"
        
        if key in self.active_signals:
            last_signal = self.active_signals[key]["signal"]
            time_diff = datetime.now() - self.active_signals[key]["timestamp"]
            
            # If same signal type and within 5 minutes, consider duplicate
            if (last_signal.signal_type == signal.signal_type and 
                time_diff < timedelta(minutes=5)):
                return True
        
        return False
    
    async def _validate_price(self, signal: PineScriptSignal) -> bool:
        """Validate that the signal price is reasonable"""
        try:
            # Get current market price from database
            async with get_async_db() as session:
                from ..src.database.queries import TimescaleQueries
                
                latest_data = await TimescaleQueries.get_latest_market_data(
                    session, signal.symbol, limit=1
                )
                
                if not latest_data:
                    return True  # No data to compare against
                
                current_price = latest_data[0]["close"]
                price_diff = abs(signal.price - current_price) / current_price
                
                # Allow 5% price difference
                return price_diff <= 0.05
                
        except Exception as e:
            logger.error(f"Error validating price: {e}")
            return True  # Allow if validation fails
    
    async def get_active_signals(self, symbol: Optional[str] = None) -> List[PineScriptSignal]:
        """Get active signals, optionally filtered by symbol"""
        active_signals = []
        current_time = datetime.now()
        
        # Clean up old signals
        expired_keys = []
        for key, signal_info in self.active_signals.items():
            if current_time - signal_info["timestamp"] > timedelta(seconds=self.signal_timeout):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.active_signals[key]
        
        # Return active signals
        for signal_info in self.active_signals.values():
            signal = signal_info["signal"]
            if symbol is None or signal.symbol == symbol:
                active_signals.append(signal)
        
        return active_signals
    
    async def get_signal_summary(self) -> Dict:
        """Get summary of active signals"""
        active_signals = await self.get_active_signals()
        
        summary = {
            "total_signals": len(active_signals),
            "buy_signals": len([s for s in active_signals if s.signal_type == SignalType.BUY]),
            "sell_signals": len([s for s in active_signals if s.signal_type == SignalType.SELL]),
            "strong_signals": len([s for s in active_signals if s.strength == SignalStrength.STRONG]),
            "medium_signals": len([s for s in active_signals if s.strength == SignalStrength.MEDIUM]),
            "weak_signals": len([s for s in active_signals if s.strength == SignalStrength.WEAK]),
            "signals_by_symbol": {},
            "signals_by_strategy": {}
        }
        
        # Group by symbol
        for signal in active_signals:
            if signal.symbol not in summary["signals_by_symbol"]:
                summary["signals_by_symbol"][signal.symbol] = []
            summary["signals_by_symbol"][signal.symbol].append({
                "type": signal.signal_type.value,
                "strength": signal.strength.value,
                "price": signal.price,
                "strategy": signal.strategy_name,
                "confidence": signal.confidence
            })
        
        # Group by strategy
        for signal in active_signals:
            if signal.strategy_name not in summary["signals_by_strategy"]:
                summary["signals_by_strategy"][signal.strategy_name] = []
            summary["signals_by_strategy"][signal.strategy_name].append({
                "symbol": signal.symbol,
                "type": signal.signal_type.value,
                "strength": signal.strength.value,
                "price": signal.price,
                "confidence": signal.confidence
            })
        
        return summary

class PineScriptWebhookHandler:
    """Handle webhook calls from TradingView Pine Script"""
    
    def __init__(self, processor: PineScriptProcessor):
        self.processor = processor
    
    async def handle_webhook(self, webhook_data: Dict) -> Dict:
        """Handle incoming webhook from TradingView"""
        try:
            # Extract signal data from webhook
            signal_data = self._extract_signal_from_webhook(webhook_data)
            
            if not signal_data:
                return {"status": "error", "message": "Invalid webhook data"}
            
            # Process signal
            signal = await self.processor.process_signal(signal_data)
            
            if signal:
                return {
                    "status": "success",
                    "message": "Signal processed successfully",
                    "signal": {
                        "symbol": signal.symbol,
                        "type": signal.signal_type.value,
                        "strength": signal.strength.value,
                        "price": signal.price,
                        "strategy": signal.strategy_name
                    }
                }
            else:
                return {"status": "error", "message": "Signal processing failed"}
                
        except Exception as e:
            logger.error(f"Error handling webhook: {e}")
            return {"status": "error", "message": str(e)}
    
    def _extract_signal_from_webhook(self, webhook_data: Dict) -> Optional[Dict]:
        """Extract signal data from TradingView webhook"""
        try:
            # TradingView webhook format
            if "symbol" in webhook_data and "signal" in webhook_data:
                return {
                    "symbol": webhook_data["symbol"],
                    "signal_type": webhook_data["signal"].lower(),
                    "price": float(webhook_data.get("price", 0)),
                    "strategy_name": webhook_data.get("strategy", "unknown"),
                    "strength": webhook_data.get("strength", "medium"),
                    "confidence": float(webhook_data.get("confidence", 0.5)),
                    "timestamp": webhook_data.get("timestamp", datetime.now().isoformat()),
                    "metadata": webhook_data.get("metadata", {})
                }
            
            # Alternative format
            if "data" in webhook_data:
                data = webhook_data["data"]
                return {
                    "symbol": data.get("symbol", ""),
                    "signal_type": data.get("action", "hold").lower(),
                    "price": float(data.get("price", 0)),
                    "strategy_name": data.get("indicator", "unknown"),
                    "strength": data.get("strength", "medium"),
                    "confidence": float(data.get("confidence", 0.5)),
                    "timestamp": data.get("time", datetime.now().isoformat()),
                    "metadata": data.get("extra", {})
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting signal from webhook: {e}")
            return None

# Global instances
pine_processor = PineScriptProcessor()
webhook_handler = PineScriptWebhookHandler(pine_processor)
