from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class MarketRegime(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"


@dataclass
class Signal:
    signal_type: SignalType
    symbol: str
    price: float
    timestamp: datetime
    confidence: float
    strategy_name: str
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    Implements common functionality and defines interface for strategy-specific logic.
    """
    
    def __init__(self, name: str, parameters: Dict = None):
        self.name = name
        self.parameters = parameters or {}
        self.is_active = True
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        """
        Generate trading signals based on market data.
        
        Args:
            data: DataFrame with OHLCV data and technical indicators
            symbol: Trading symbol
            
        Returns:
            List of Signal objects
        """
        pass
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with calculated indicators
        """
        pass
    
    def validate_signal(self, signal: Signal, market_regime: MarketRegime) -> bool:
        """
        Validate if a signal meets the strategy's criteria.
        
        Args:
            signal: Signal to validate
            market_regime: Current market regime
            
        Returns:
            True if signal is valid, False otherwise
        """
        # Base validation - can be overridden by subclasses
        if signal.confidence < 0.5:
            return False
        
        # Market regime specific validation
        if market_regime == MarketRegime.VOLATILE:
            # Require higher confidence in volatile markets
            if signal.confidence < 0.7:
                return False
        
        return True
    
    def calculate_position_size(self, signal: Signal, account_balance: float, 
                              risk_per_trade: float = 0.02) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            signal: Trading signal
            account_balance: Current account balance
            risk_per_trade: Percentage of account to risk per trade
            
        Returns:
            Position size in base currency
        """
        risk_amount = account_balance * risk_per_trade
        # This is a simplified calculation - in practice, you'd use stop loss distance
        position_size = risk_amount / signal.price
        return position_size
    
    def calculate_stop_loss(self, signal: Signal, atr: float, 
                           atr_multiplier: float = 2.0) -> float:
        """
        Calculate dynamic stop loss based on ATR.
        
        Args:
            signal: Trading signal
            atr: Average True Range
            atr_multiplier: Multiplier for ATR
            
        Returns:
            Stop loss price
        """
        if signal.signal_type == SignalType.BUY:
            return signal.price - (atr * atr_multiplier)
        else:
            return signal.price + (atr * atr_multiplier)
    
    def calculate_take_profit(self, signal: Signal, risk_reward_ratio: float = 2.0) -> float:
        """
        Calculate take profit based on risk:reward ratio.
        
        Args:
            signal: Trading signal
            risk_reward_ratio: Desired risk:reward ratio
            
        Returns:
            Take profit price
        """
        # This would need the stop loss to calculate properly
        # For now, return a simple calculation
        if signal.signal_type == SignalType.BUY:
            return signal.price * (1 + (risk_reward_ratio * 0.01))
        else:
            return signal.price * (1 - (risk_reward_ratio * 0.01))
    
    def get_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """
        Determine current market regime based on price action and volatility.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            MarketRegime enum value
        """
        if len(data) < 20:
            return MarketRegime.RANGING
        
        # Calculate volatility (ATR percentage)
        if 'atr' in data.columns and 'close' in data.columns:
            current_atr = data['atr'].iloc[-1]
            current_price = data['close'].iloc[-1]
            volatility = current_atr / current_price
            
            # High volatility
            if volatility > 0.03:
                return MarketRegime.VOLATILE
            
            # Check for trending vs ranging
            if 'ema_50' in data.columns and 'ema_200' in data.columns:
                ema_50 = data['ema_50'].iloc[-1]
                ema_200 = data['ema_200'].iloc[-1]
                price = data['close'].iloc[-1]
                
                # Strong trend if price is far from EMAs
                ema_distance = abs(price - ema_50) / ema_50
                if ema_distance > 0.02:
                    return MarketRegime.TRENDING
        
        return MarketRegime.RANGING
    
    def filter_signals_by_timeframe(self, signals: List[Signal], 
                                  required_timeframes: List[str]) -> List[Signal]:
        """
        Filter signals to ensure they align across multiple timeframes.
        
        Args:
            signals: List of signals from different timeframes
            required_timeframes: List of timeframes that must agree
            
        Returns:
            Filtered list of signals
        """
        # Group signals by symbol
        signals_by_symbol = {}
        for signal in signals:
            if signal.symbol not in signals_by_symbol:
                signals_by_symbol[signal.symbol] = []
            signals_by_symbol[signal.symbol].append(signal)
        
        # Check for confluence across timeframes
        filtered_signals = []
        for symbol, symbol_signals in signals_by_symbol.items():
            # Count signals by type
            buy_signals = [s for s in symbol_signals if s.signal_type == SignalType.BUY]
            sell_signals = [s for s in symbol_signals if s.signal_type == SignalType.SELL]
            
            # Require at least 2 timeframes to agree
            if len(buy_signals) >= 2:
                # Use the signal with highest confidence
                best_buy = max(buy_signals, key=lambda x: x.confidence)
                filtered_signals.append(best_buy)
            elif len(sell_signals) >= 2:
                best_sell = max(sell_signals, key=lambda x: x.confidence)
                filtered_signals.append(best_sell)
        
        return filtered_signals
    
    def update_performance_metrics(self, trade_results: List[Dict]):
        """
        Update strategy performance metrics based on trade results.
        
        Args:
            trade_results: List of trade result dictionaries
        """
        if not trade_results:
            return
        
        total_trades = len(trade_results)
        winning_trades = len([t for t in trade_results if t.get('pnl', 0) > 0])
        losing_trades = total_trades - winning_trades
        
        self.parameters.update({
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'avg_profit': np.mean([t.get('pnl', 0) for t in trade_results if t.get('pnl', 0) > 0]) if winning_trades > 0 else 0,
            'avg_loss': np.mean([t.get('pnl', 0) for t in trade_results if t.get('pnl', 0) < 0]) if losing_trades > 0 else 0
        })
    
    def __str__(self):
        return f"{self.name} Strategy"
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(name='{self.name}')>"
