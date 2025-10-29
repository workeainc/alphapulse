import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .pattern_detector import CandlestickPatternDetector
from .indicators import TechnicalIndicators
from ..src.database.models import PatternStats
from ..src.database.connection import get_db

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"

class VolumeCondition(Enum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"

@dataclass
class PatternSignal:
    pattern_name: str
    timestamp: datetime
    price: float
    signal_type: str  # "bullish" or "bearish"
    confidence: float
    market_regime: MarketRegime
    volume_condition: VolumeCondition
    metadata: Dict

@dataclass
class PatternResult:
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    pnl: float
    pnl_percentage: float
    holding_period: timedelta
    success: bool

class PatternBacktester:
    """
    Backtesting framework for candlestick pattern performance analysis
    """
    
    def __init__(self, risk_per_trade: float = 0.02, max_holding_period: int = 48):
        self.risk_per_trade = risk_per_trade  # 2% risk per trade
        self.max_holding_period = max_holding_period  # Maximum holding period in hours
        self.pattern_detector = CandlestickPatternDetector()
        self.indicators = TechnicalIndicators()
        
    async def backtest_pattern(
        self, 
        df: pd.DataFrame, 
        pattern_name: str, 
        symbol: str, 
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> PatternStats:
        """
        Backtest a specific pattern and return performance statistics
        """
        logger.info(f"🔍 Backtesting {pattern_name} for {symbol} {timeframe}")
        
        # Filter data by date range if provided
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        if len(df) < 100:
            logger.warning(f"Insufficient data for backtesting: {len(df)} rows")
            return None
        
        # Detect patterns
        pattern_signals = self._detect_pattern_signals(df, pattern_name)
        
        if not pattern_signals:
            logger.warning(f"No {pattern_name} signals detected")
            return None
        
        # Simulate trades
        trade_results = self._simulate_trades(df, pattern_signals)
        
        if not trade_results:
            logger.warning(f"No valid trades for {pattern_name}")
            return None
        
        # Calculate performance metrics
        stats = self._calculate_performance_metrics(trade_results, pattern_name, symbol, timeframe)
        
        logger.info(f"✅ {pattern_name} backtest completed: {stats.win_rate:.2%} win rate, {stats.avg_rr:.2f} avg R/R")
        
        return stats
    
    def _detect_pattern_signals(self, df: pd.DataFrame, pattern_name: str) -> List[PatternSignal]:
        """Detect pattern signals and classify market conditions"""
        signals = []
        
        # Add technical indicators for market regime detection
        df = self.indicators.add_all_indicators(df)
        
        for i in range(50, len(df) - 1):  # Skip first 50 candles for indicator stability
            try:
                # Detect pattern
                pattern_result = self.pattern_detector.detect_single_pattern(
                    df.iloc[:i+1], pattern_name
                )
                
                if pattern_result and pattern_result['detected']:
                    # Classify market regime
                    market_regime = self._classify_market_regime(df.iloc[:i+1])
                    
                    # Classify volume condition
                    volume_condition = self._classify_volume_condition(df.iloc[:i+1])
                    
                    signal = PatternSignal(
                        pattern_name=pattern_name,
                        timestamp=df.index[i],
                        price=df.iloc[i]['close'],
                        signal_type=pattern_result['type'],
                        confidence=pattern_result.get('strength', 0.5),
                        market_regime=market_regime,
                        volume_condition=volume_condition,
                        metadata=pattern_result.get('metadata', {})
                    )
                    
                    signals.append(signal)
                    
            except Exception as e:
                logger.debug(f"Error detecting pattern at index {i}: {e}")
                continue
        
        return signals
    
    def _classify_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Classify current market regime based on technical indicators"""
        if len(df) < 20:
            return MarketRegime.RANGING
        
        # Get latest values
        current = df.iloc[-1]
        
        # ADX for trend strength
        adx = current.get('adx', 0)
        
        # ATR ratio for volatility
        atr = current.get('atr', 0)
        atr_sma = df['atr'].rolling(20).mean().iloc[-1] if 'atr' in df.columns else atr
        atr_ratio = atr / atr_sma if atr_sma > 0 else 1.0
        
        # Price vs EMA for trend direction
        ema_20 = current.get('ema_20', current['close'])
        ema_50 = current.get('ema_50', current['close'])
        price_vs_ema = current['close'] / ema_20 if ema_20 > 0 else 1.0
        
        # Classify regime
        if adx > 25 and abs(price_vs_ema - 1.0) > 0.02:
            return MarketRegime.TRENDING
        elif atr_ratio > 1.3:
            return MarketRegime.VOLATILE
        else:
            return MarketRegime.RANGING
    
    def _classify_volume_condition(self, df: pd.DataFrame) -> VolumeCondition:
        """Classify volume condition based on recent volume patterns"""
        if len(df) < 20:
            return VolumeCondition.NORMAL
        
        current_volume = df.iloc[-1]['volume']
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        if volume_ratio > 1.5:
            return VolumeCondition.HIGH
        elif volume_ratio < 0.7:
            return VolumeCondition.LOW
        else:
            return VolumeCondition.NORMAL
    
    def _simulate_trades(self, df: pd.DataFrame, signals: List[PatternSignal]) -> List[PatternResult]:
        """Simulate trades based on pattern signals"""
        results = []
        
        for signal in signals:
            try:
                # Find entry point
                entry_idx = df.index.get_loc(signal.timestamp)
                if entry_idx >= len(df) - 1:
                    continue
                
                entry_price = signal.price
                entry_time = signal.timestamp
                
                # Calculate stop loss and take profit
                atr = df.iloc[entry_idx].get('atr', entry_price * 0.02)
                
                if signal.signal_type == "bullish":
                    stop_loss = entry_price - (atr * 2)  # 2 ATR stop loss
                    take_profit = entry_price + (atr * 4)  # 4 ATR take profit
                else:
                    stop_loss = entry_price + (atr * 2)
                    take_profit = entry_price - (atr * 4)
                
                # Simulate trade exit
                exit_result = self._simulate_trade_exit(
                    df.iloc[entry_idx+1:], 
                    entry_price, 
                    stop_loss, 
                    take_profit, 
                    signal.signal_type,
                    entry_time
                )
                
                if exit_result:
                    results.append(exit_result)
                    
            except Exception as e:
                logger.debug(f"Error simulating trade for signal {signal.timestamp}: {e}")
                continue
        
        return results
    
    def _simulate_trade_exit(
        self, 
        future_data: pd.DataFrame, 
        entry_price: float, 
        stop_loss: float, 
        take_profit: float, 
        signal_type: str,
        entry_time: datetime
    ) -> Optional[PatternResult]:
        """Simulate trade exit based on stop loss, take profit, or max holding period"""
        
        for i, (timestamp, candle) in enumerate(future_data.iterrows()):
            # Check if max holding period reached
            if i >= self.max_holding_period:
                exit_price = candle['close']
                exit_time = timestamp
                pnl = (exit_price - entry_price) if signal_type == "bullish" else (entry_price - exit_price)
                pnl_percentage = pnl / entry_price
                success = pnl > 0
                
                return PatternResult(
                    entry_price=entry_price,
                    entry_time=entry_time,
                    exit_price=exit_price,
                    exit_time=exit_time,
                    pnl=pnl,
                    pnl_percentage=pnl_percentage,
                    holding_period=exit_time - entry_time,
                    success=success
                )
            
            # Check stop loss
            if signal_type == "bullish":
                if candle['low'] <= stop_loss:
                    exit_price = stop_loss
                    exit_time = timestamp
                    pnl = exit_price - entry_price
                    pnl_percentage = pnl / entry_price
                    success = False
                    
                    return PatternResult(
                        entry_price=entry_price,
                        entry_time=entry_time,
                        exit_price=exit_price,
                        exit_time=exit_time,
                        pnl=pnl,
                        pnl_percentage=pnl_percentage,
                        holding_period=exit_time - entry_time,
                        success=success
                    )
                
                # Check take profit
                if candle['high'] >= take_profit:
                    exit_price = take_profit
                    exit_time = timestamp
                    pnl = exit_price - entry_price
                    pnl_percentage = pnl / entry_price
                    success = True
                    
                    return PatternResult(
                        entry_price=entry_price,
                        entry_time=entry_time,
                        exit_price=exit_price,
                        exit_time=exit_time,
                        pnl=pnl,
                        pnl_percentage=pnl_percentage,
                        holding_period=exit_time - entry_time,
                        success=success
                    )
            
            else:  # Bearish signal
                if candle['high'] >= stop_loss:
                    exit_price = stop_loss
                    exit_time = timestamp
                    pnl = entry_price - exit_price
                    pnl_percentage = pnl / entry_price
                    success = False
                    
                    return PatternResult(
                        entry_price=entry_price,
                        entry_time=entry_time,
                        exit_price=exit_price,
                        exit_time=exit_time,
                        pnl=pnl,
                        pnl_percentage=pnl_percentage,
                        holding_period=exit_time - entry_time,
                        success=success
                    )
                
                # Check take profit
                if candle['low'] <= take_profit:
                    exit_price = take_profit
                    exit_time = timestamp
                    pnl = entry_price - exit_price
                    pnl_percentage = pnl / entry_price
                    success = True
                    
                    return PatternResult(
                        entry_price=entry_price,
                        entry_time=entry_time,
                        exit_price=exit_price,
                        exit_time=exit_time,
                        pnl=pnl,
                        pnl_percentage=pnl_percentage,
                        holding_period=exit_time - entry_time,
                        success=success
                    )
        
        return None
    
    def _calculate_performance_metrics(
        self, 
        results: List[PatternResult], 
        pattern_name: str, 
        symbol: str, 
        timeframe: str
    ) -> PatternStats:
        """Calculate comprehensive performance metrics from trade results"""
        
        if not results:
            return None
        
        total_trades = len(results)
        winning_trades = len([r for r in results if r.success])
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate profit/loss metrics
        profits = [r.pnl for r in results if r.success]
        losses = [r.pnl for r in results if not r.success]
        
        avg_profit = np.mean(profits) if profits else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        # Calculate risk/reward ratio
        if avg_loss != 0:
            avg_rr = abs(avg_profit / avg_loss)
        else:
            avg_rr = 0.0
        
        # Calculate profit factor
        gross_profit = sum(profits) if profits else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Calculate historical success factor (normalized)
        historical_success_factor = min(win_rate * avg_rr, 2.0)  # Cap at 2.0
        
        # Determine best market conditions (simplified for now)
        best_market_regime = "trending"  # Default, will be enhanced in Phase 2
        best_volume_conditions = "high"  # Default, will be enhanced in Phase 2
        
        return PatternStats(
            pattern_name=pattern_name,
            symbol=symbol,
            timeframe=timeframe,
            total_signals=total_trades,
            winning_signals=winning_trades,
            losing_signals=losing_trades,
            win_rate=win_rate,
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            avg_rr=avg_rr,
            profit_factor=profit_factor,
            historical_success_factor=historical_success_factor,
            sample_size=total_trades,
            min_sample_size=10,
            start_date=min(r.entry_time for r in results),
            end_date=max(r.exit_time for r in results),
            best_market_regime=best_market_regime,
            best_volume_conditions=best_volume_conditions
        )
    
    async def save_pattern_stats(self, stats: PatternStats) -> bool:
        """Save pattern statistics to database"""
        try:
            async with get_db() as session:
                # Check if stats already exist for this pattern/symbol/timeframe
                existing = await session.execute(
                    "SELECT id FROM pattern_stats WHERE pattern_name = :pattern_name AND symbol = :symbol AND timeframe = :timeframe",
                    {"pattern_name": stats.pattern_name, "symbol": stats.symbol, "timeframe": stats.timeframe}
                )
                
                if existing.fetchone():
                    # Update existing record
                    await session.execute(
                        """
                        UPDATE pattern_stats SET
                            total_signals = :total_signals,
                            winning_signals = :winning_signals,
                            losing_signals = :losing_signals,
                            win_rate = :win_rate,
                            avg_profit = :avg_profit,
                            avg_loss = :avg_loss,
                            avg_rr = :avg_rr,
                            profit_factor = :profit_factor,
                            historical_success_factor = :historical_success_factor,
                            sample_size = :sample_size,
                            start_date = :start_date,
                            end_date = :end_date,
                            best_market_regime = :best_market_regime,
                            best_volume_conditions = :best_volume_conditions,
                            last_updated = NOW()
                        WHERE pattern_name = :pattern_name AND symbol = :symbol AND timeframe = :timeframe
                        """,
                        {
                            "pattern_name": stats.pattern_name,
                            "symbol": stats.symbol,
                            "timeframe": stats.timeframe,
                            "total_signals": stats.total_signals,
                            "winning_signals": stats.winning_signals,
                            "losing_signals": stats.losing_signals,
                            "win_rate": stats.win_rate,
                            "avg_profit": stats.avg_profit,
                            "avg_loss": stats.avg_loss,
                            "avg_rr": stats.avg_rr,
                            "profit_factor": stats.profit_factor,
                            "historical_success_factor": stats.historical_success_factor,
                            "sample_size": stats.sample_size,
                            "start_date": stats.start_date,
                            "end_date": stats.end_date,
                            "best_market_regime": stats.best_market_regime,
                            "best_volume_conditions": stats.best_volume_conditions
                        }
                    )
                else:
                    # Insert new record
                    await session.execute(
                        """
                        INSERT INTO pattern_stats (
                            pattern_name, symbol, timeframe, total_signals, winning_signals,
                            losing_signals, win_rate, avg_profit, avg_loss, avg_rr,
                            profit_factor, historical_success_factor, sample_size,
                            start_date, end_date, best_market_regime, best_volume_conditions
                        ) VALUES (
                            :pattern_name, :symbol, :timeframe, :total_signals, :winning_signals,
                            :losing_signals, :win_rate, :avg_profit, :avg_loss, :avg_rr,
                            :profit_factor, :historical_success_factor, :sample_size,
                            :start_date, :end_date, :best_market_regime, :best_volume_conditions
                        )
                        """,
                        {
                            "pattern_name": stats.pattern_name,
                            "symbol": stats.symbol,
                            "timeframe": stats.timeframe,
                            "total_signals": stats.total_signals,
                            "winning_signals": stats.winning_signals,
                            "losing_signals": stats.losing_signals,
                            "win_rate": stats.win_rate,
                            "avg_profit": stats.avg_profit,
                            "avg_loss": stats.avg_loss,
                            "avg_rr": stats.avg_rr,
                            "profit_factor": stats.profit_factor,
                            "historical_success_factor": stats.historical_success_factor,
                            "sample_size": stats.sample_size,
                            "start_date": stats.start_date,
                            "end_date": stats.end_date,
                            "best_market_regime": stats.best_market_regime,
                            "best_volume_conditions": stats.best_volume_conditions
                        }
                    )
                
                await session.commit()
                logger.info(f"✅ Pattern stats saved for {stats.pattern_name} {stats.symbol} {stats.timeframe}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error saving pattern stats: {e}")
            return False
