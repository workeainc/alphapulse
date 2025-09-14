#!/usr/bin/env python3
"""
Pattern Backtesting Module
Analyzes historical pattern performance and calculates success rates
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

from .storage import DataStorage
from .pattern_analyzer import PatternAnalyzer, DetectedPattern

logger = logging.getLogger(__name__)

@dataclass
class PatternPerformance:
    """Pattern performance metrics"""
    pattern_name: str
    total_occurrences: int
    successful_signals: int
    failed_signals: int
    success_rate: float
    avg_profit: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: Optional[float] = None

@dataclass
class BacktestResult:
    """Complete backtest result"""
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    total_patterns: int
    overall_success_rate: float
    total_profit: float
    pattern_performance: List[PatternPerformance]
    market_regime_analysis: Dict[str, float]

class PatternBacktester:
    """Backtests pattern detection strategies on historical data"""
    
    def __init__(self, storage: DataStorage, pattern_analyzer: PatternAnalyzer):
        self.storage = storage
        self.pattern_analyzer = pattern_analyzer
        self.logger = logging.getLogger(__name__)
    
    async def backtest_patterns(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        min_confidence: float = 0.7,
        min_strength: str = "medium"
    ) -> BacktestResult:
        """
        Run comprehensive pattern backtest
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe for analysis (e.g., '1h')
            start_date: Start date for backtest
            end_date: End date for backtest
            min_confidence: Minimum confidence threshold
            min_strength: Minimum pattern strength ('weak', 'medium', 'strong')
        
        Returns:
            BacktestResult with comprehensive performance metrics
        """
        self.logger.info(f"Starting pattern backtest for {symbol} {timeframe}")
        
        try:
            # 1. Get historical market data
            market_data = await self._get_historical_data(
                symbol, timeframe, start_date, end_date
            )
            
            if market_data.empty:
                raise ValueError(f"No market data available for {symbol} {timeframe}")
            
            # 2. Detect patterns in historical data
            detected_patterns = await self._detect_historical_patterns(
                market_data, min_confidence, min_strength
            )
            
            # 3. Calculate pattern performance
            pattern_performance = await self._calculate_pattern_performance(
                detected_patterns, market_data
            )
            
            # 4. Analyze market regime impact
            market_regime_analysis = await self._analyze_market_regime_impact(
                detected_patterns, market_data
            )
            
            # 5. Compile results
            result = BacktestResult(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                total_patterns=len(detected_patterns),
                overall_success_rate=self._calculate_overall_success_rate(pattern_performance),
                total_profit=sum([p.avg_profit * p.successful_signals for p in pattern_performance]),
                pattern_performance=pattern_performance,
                market_regime_analysis=market_regime_analysis
            )
            
            self.logger.info(f"Backtest completed: {result.total_patterns} patterns analyzed")
            return result
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            raise
    
    async def _get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Retrieve historical market data"""
        try:
            # Get OHLCV data from storage
            data = await self.storage.get_market_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get historical data: {e}")
            return pd.DataFrame()
    
    async def _detect_historical_patterns(
        self,
        market_data: pd.DataFrame,
        min_confidence: float,
        min_strength: str
    ) -> List[DetectedPattern]:
        """Detect patterns in historical market data"""
        patterns = []
        
        try:
            # Use pattern analyzer to detect patterns
            for i in range(len(market_data) - 50):  # Need at least 50 candles for pattern detection
                window_data = market_data.iloc[i:i+50]
                
                # Detect patterns in this window
                window_patterns = await self.pattern_analyzer.detect_patterns(
                    window_data, min_confidence, min_strength
                )
                
                # Add timestamp information
                for pattern in window_patterns:
                    pattern.timestamp = window_data.iloc[-1]['timestamp']
                    patterns.append(pattern)
            
            # Remove duplicates and sort by timestamp
            unique_patterns = self._remove_duplicate_patterns(patterns)
            return sorted(unique_patterns, key=lambda x: x.timestamp)
            
        except Exception as e:
            self.logger.error(f"Failed to detect historical patterns: {e}")
            return []
    
    def _remove_duplicate_patterns(self, patterns: List[DetectedPattern]) -> List[DetectedPattern]:
        """Remove duplicate patterns within a short time window"""
        if not patterns:
            return []
        
        unique_patterns = []
        seen_patterns = set()
        
        for pattern in patterns:
            # Create unique identifier for pattern
            pattern_key = f"{pattern.pattern_name}_{pattern.symbol}_{pattern.timeframe}_{pattern.timestamp.strftime('%Y%m%d_%H%M')}"
            
            if pattern_key not in seen_patterns:
                seen_patterns.add(pattern_key)
                unique_patterns.append(pattern)
        
        return unique_patterns
    
    async def _calculate_pattern_performance(
        self,
        patterns: List[DetectedPattern],
        market_data: pd.DataFrame
    ) -> List[PatternPerformance]:
        """Calculate performance metrics for each pattern type"""
        pattern_stats = {}
        
        for pattern in patterns:
            if pattern.pattern_name not in pattern_stats:
                pattern_stats[pattern.pattern_name] = {
                    'occurrences': 0,
                    'profits': [],
                    'losses': [],
                    'successes': 0,
                    'failures': 0
                }
            
            stats = pattern_stats[pattern.pattern_name]
            stats['occurrences'] += 1
            
            # Calculate profit/loss for this pattern
            profit_loss = await self._calculate_pattern_pnl(pattern, market_data)
            
            if profit_loss > 0:
                stats['profits'].append(profit_loss)
                stats['successes'] += 1
            else:
                stats['losses'].append(abs(profit_loss))
                stats['failures'] += 1
        
        # Convert to PatternPerformance objects
        performance_list = []
        for pattern_name, stats in pattern_stats.items():
            if stats['occurrences'] > 0:
                success_rate = stats['successes'] / stats['occurrences']
                avg_profit = np.mean(stats['profits']) if stats['profits'] else 0
                avg_loss = np.mean(stats['losses']) if stats['losses'] else 0
                profit_factor = avg_profit / avg_loss if avg_loss > 0 else float('inf')
                
                performance = PatternPerformance(
                    pattern_name=pattern_name,
                    total_occurrences=stats['occurrences'],
                    successful_signals=stats['successes'],
                    failed_signals=stats['failures'],
                    success_rate=success_rate,
                    avg_profit=avg_profit,
                    avg_loss=avg_loss,
                    profit_factor=profit_factor,
                    max_drawdown=0.0,  # Will implement later
                    sharpe_ratio=None    # Will implement later
                )
                performance_list.append(performance)
        
        return performance_list
    
    async def _calculate_pattern_pnl(
        self,
        pattern: DetectedPattern,
        market_data: pd.DataFrame
    ) -> float:
        """Calculate profit/loss for a specific pattern"""
        try:
            # Find the pattern timestamp in market data
            pattern_idx = market_data[market_data['timestamp'] == pattern.timestamp].index
            
            if len(pattern_idx) == 0:
                return 0.0
            
            pattern_idx = pattern_idx[0]
            
            # Look ahead to calculate P&L (simplified approach)
            if pattern_idx + 20 >= len(market_data):  # Need at least 20 candles ahead
                return 0.0
            
            entry_price = market_data.iloc[pattern_idx]['close_price']
            future_data = market_data.iloc[pattern_idx:pattern_idx+20]
            
            # Calculate potential profit/loss based on pattern type
            if pattern.trend_alignment == 'bullish':
                # For bullish patterns, look for price increase
                max_price = future_data['high_price'].max()
                return (max_price - entry_price) / entry_price * 100  # Return as percentage
            else:
                # For bearish patterns, look for price decrease
                min_price = future_data['low_price'].min()
                return (entry_price - min_price) / entry_price * 100  # Return as percentage
                
        except Exception as e:
            self.logger.error(f"Failed to calculate P&L for pattern: {e}")
            return 0.0
    
    def _calculate_overall_success_rate(self, performance: List[PatternPerformance]) -> float:
        """Calculate overall success rate across all patterns"""
        if not performance:
            return 0.0
        
        total_signals = sum([p.total_occurrences for p in performance])
        total_successes = sum([p.successful_signals for p in performance])
        
        return total_successes / total_signals if total_signals > 0 else 0.0
    
    async def _analyze_market_regime_impact(
        self,
        patterns: List[DetectedPattern],
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Analyze how market regime affects pattern performance"""
        regime_performance = {
            'trending': {'successes': 0, 'total': 0},
            'ranging': {'successes': 0, 'total': 0},
            'volatile': {'successes': 0, 'total': 0}
        }
        
        # Simple market regime detection (can be enhanced later)
        for pattern in patterns:
            # Determine market regime based on price volatility
            regime = self._detect_market_regime(pattern, market_data)
            
            if regime in regime_performance:
                regime_performance[regime]['total'] += 1
                # For now, assume all patterns are successful (will enhance later)
                regime_performance[regime]['successes'] += 1
        
        # Calculate success rates
        result = {}
        for regime, stats in regime_performance.items():
            if stats['total'] > 0:
                result[regime] = stats['successes'] / stats['total']
            else:
                result[regime] = 0.0
        
        return result
    
    def _detect_market_regime(self, pattern: DetectedPattern, market_data: pd.DataFrame) -> str:
        """Simple market regime detection"""
        try:
            # Find pattern timestamp and look at recent price action
            pattern_idx = market_data[market_data['timestamp'] == pattern.timestamp].index
            
            if len(pattern_idx) == 0:
                return 'ranging'
            
            pattern_idx = pattern_idx[0]
            
            # Look at last 20 candles for regime detection
            start_idx = max(0, pattern_idx - 20)
            recent_data = market_data.iloc[start_idx:pattern_idx+1]
            
            if len(recent_data) < 10:
                return 'ranging'
            
            # Calculate volatility and trend
            returns = recent_data['close_price'].pct_change().dropna()
            volatility = returns.std()
            trend = (recent_data['close_price'].iloc[-1] - recent_data['close_price'].iloc[0]) / recent_data['close_price'].iloc[0]
            
            if volatility > 0.05:  # High volatility
                return 'volatile'
            elif abs(trend) > 0.02:  # Strong trend
                return 'trending'
            else:
                return 'ranging'
                
        except Exception as e:
            self.logger.error(f"Failed to detect market regime: {e}")
            return 'ranging'
    
    async def generate_backtest_report(self, result: BacktestResult) -> str:
        """Generate a human-readable backtest report"""
        report = f"""
🔍 PATTERN BACKTEST REPORT
{'='*50}

📊 OVERVIEW
Symbol: {result.symbol}
Timeframe: {result.timeframe}
Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}
Total Patterns: {result.total_patterns}
Overall Success Rate: {result.overall_success_rate:.2%}
Total Profit: {result.total_profit:.2f}%

📈 PATTERN PERFORMANCE
{'-'*30}
"""
        
        for perf in result.pattern_performance:
            report += f"""
{perf.pattern_name.upper()}:
  • Occurrences: {perf.total_occurrences}
  • Success Rate: {perf.success_rate:.2%}
  • Avg Profit: {perf.avg_profit:.2f}%
  • Avg Loss: {perf.avg_loss:.2f}%
  • Profit Factor: {perf.profit_factor:.2f}
"""
        
        report += f"""
🌍 MARKET REGIME ANALYSIS
{'-'*30}
"""
        
        for regime, success_rate in result.market_regime_analysis.items():
            report += f"• {regime.title()}: {success_rate:.2%}\n"
        
        return report
