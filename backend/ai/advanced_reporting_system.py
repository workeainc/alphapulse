"""
Advanced Reporting System for AlphaPulse
Provides comprehensive trading reports, performance analytics, and market insights
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import json
import numpy as np
import pandas as pd

# Import our components
try:
    from ..ai.advanced_analytics_engine import AdvancedAnalyticsEngine
    from ..database.connection import TimescaleDBConnection
    from ..core.trading_engine import TradingEngine
except ImportError as e:
    logging.warning(f"Some imports not available: {e}")
    AdvancedAnalyticsEngine = None
    TimescaleDBConnection = None
    TradingEngine = None

logger = logging.getLogger(__name__)

@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    symbol: str
    timestamp: datetime
    period: str  # 'daily', 'weekly', 'monthly', 'custom'
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    metadata: Dict[str, Any]

@dataclass
class RiskReport:
    """Comprehensive risk report"""
    symbol: str
    timestamp: datetime
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    volatility: float
    beta: float
    correlation: float
    concentration_risk: float
    liquidity_risk: float
    metadata: Dict[str, Any]

@dataclass
class MarketReport:
    """Comprehensive market report"""
    symbol: str
    timestamp: datetime
    market_regime: str
    regime_confidence: float
    regime_duration: timedelta
    price_trend: str
    volume_trend: str
    volatility_regime: str
    support_levels: List[float]
    resistance_levels: List[float]
    key_insights: List[str]
    metadata: Dict[str, Any]

class AdvancedReportingSystem:
    """Advanced reporting system for comprehensive trading analysis"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Reporting configuration
        self.report_frequency = self.config.get('report_frequency', 'daily')
        self.symbols = self.config.get('symbols', ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'])
        self.enable_auto_reports = self.config.get('enable_auto_reports', True)
        self.report_retention_days = self.config.get('report_retention_days', 90)
        
        # Component references
        self.analytics_engine = None
        self.db_connection = None
        self.trading_engine = None
        
        # Report storage
        self.performance_reports = defaultdict(deque)  # symbol -> reports
        self.risk_reports = defaultdict(deque)  # symbol -> reports
        self.market_reports = defaultdict(deque)  # symbol -> reports
        self.combined_reports = defaultdict(deque)  # symbol -> combined_reports
        
        # Performance tracking
        self.stats = {
            'total_reports_generated': 0,
            'performance_reports': 0,
            'risk_reports': 0,
            'market_reports': 0,
            'last_report': None,
            'report_generation_times': deque(maxlen=100)
        }
        
        # Callbacks
        self.report_callbacks = defaultdict(list)  # report_type -> [callback]
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize reporting components"""
        try:
            # Initialize analytics engine if available
            if AdvancedAnalyticsEngine:
                analytics_config = {
                    'symbols': self.symbols,
                    'update_frequency': 10.0,
                    'enable_predictions': True,
                    'enable_regime_detection': True
                }
                self.analytics_engine = AdvancedAnalyticsEngine(analytics_config)
                self.logger.info("Analytics engine initialized for reporting")
            
            # Initialize database connection if available
            if TimescaleDBConnection:
                db_config = self.config.get('database', {})
                self.db_connection = TimescaleDBConnection(db_config)
                self.logger.info("Database connection initialized for reporting")
            
            # Initialize trading engine if available
            if TradingEngine:
                self.trading_engine = TradingEngine()
                self.logger.info("Trading engine initialized for reporting")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize reporting components: {e}")
    
    async def initialize(self):
        """Initialize the reporting system"""
        try:
            self.logger.info("Initializing Advanced Reporting System...")
            
            # Initialize analytics engine
            if self.analytics_engine:
                await self.analytics_engine.initialize()
            
            # Initialize database connection
            if self.db_connection:
                await self.db_connection.initialize()
            
            # Initialize trading engine
            if self.trading_engine:
                await self.trading_engine.initialize()
            
            self.logger.info("Advanced Reporting System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize reporting system: {e}")
            raise
    
    async def generate_comprehensive_report(self, symbol: str, period: str = 'daily') -> Dict[str, Any]:
        """Generate comprehensive report for a symbol"""
        try:
            start_time = time.time()
            
            # Generate individual reports
            performance_report = await self._generate_performance_report(symbol, period)
            risk_report = await self._generate_risk_report(symbol, period)
            market_report = await self._generate_market_report(symbol, period)
            
            # Combine reports
            combined_report = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'period': period,
                'performance': performance_report,
                'risk': risk_report,
                'market': market_report,
                'summary': await self._generate_report_summary(performance_report, risk_report, market_report),
                'metadata': {
                    'generation_time': time.time() - start_time,
                    'report_version': '1.0',
                    'components_used': ['analytics_engine', 'database', 'trading_engine']
                }
            }
            
            # Store combined report
            self.combined_reports[symbol].append(combined_report)
            if len(self.combined_reports[symbol]) > 100:
                self.combined_reports[symbol].popleft()
            
            # Update statistics
            self.stats['total_reports_generated'] += 1
            self.stats['last_report'] = datetime.now()
            generation_time = time.time() - start_time
            self.stats['report_generation_times'].append(generation_time)
            
            # Trigger callbacks
            await self._trigger_callbacks('comprehensive_report', combined_report)
            
            return combined_report
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report for {symbol}: {e}")
            return {}
    
    async def _generate_performance_report(self, symbol: str, period: str) -> Optional[PerformanceReport]:
        """Generate performance report for a symbol"""
        try:
            # Get trading data from database
            if not self.db_connection:
                return None
            
            # Calculate period boundaries
            end_time = datetime.now()
            if period == 'daily':
                start_time = end_time - timedelta(days=1)
            elif period == 'weekly':
                start_time = end_time - timedelta(weeks=1)
            elif period == 'monthly':
                start_time = end_time - timedelta(days=30)
            else:
                start_time = end_time - timedelta(days=1)
            
            # Get trades for the period
            trades = await self.db_connection.get_trades(
                symbol=symbol,
                start_time=start_time,
                limit=1000
            )
            
            if not trades:
                return None
            
            # Calculate performance metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
            losing_trades = len([t for t in trades if t.get('pnl', 0) < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate PnL metrics
            pnls = [t.get('pnl', 0) for t in trades]
            total_pnl = sum(pnls)
            
            # Calculate drawdown
            cumulative_pnl = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdowns = running_max - cumulative_pnl
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
            
            # Calculate risk-adjusted metrics
            returns = np.diff(cumulative_pnl)
            if len(returns) > 0:
                volatility = np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
                sharpe_ratio = (np.mean(returns) * 252) / volatility if volatility > 0 else 0
                
                # Sortino ratio (downside deviation)
                downside_returns = returns[returns < 0]
                downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
                sortino_ratio = (np.mean(returns) * 252) / downside_deviation if downside_deviation > 0 else 0
                
                # Calmar ratio
                calmar_ratio = (np.mean(returns) * 252) / max_drawdown if max_drawdown > 0 else 0
            else:
                volatility = sharpe_ratio = sortino_ratio = calmar_ratio = 0
            
            # Create performance report
            report = PerformanceReport(
                symbol=symbol,
                timestamp=datetime.now(),
                period=period,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                metadata={
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'trade_ids': [t.get('id') for t in trades],
                    'calculation_method': 'standard_metrics'
                }
            )
            
            # Store report
            self.performance_reports[symbol].append(report)
            if len(self.performance_reports[symbol]) > 50:
                self.performance_reports[symbol].popleft()
            
            # Update statistics
            self.stats['performance_reports'] += 1
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report for {symbol}: {e}")
            return None
    
    async def _generate_risk_report(self, symbol: str, period: str) -> Optional[RiskReport]:
        """Generate risk report for a symbol"""
        try:
            # Get market data for risk calculation
            if not self.db_connection:
                return None
            
            # Calculate period boundaries
            end_time = datetime.now()
            if period == 'daily':
                start_time = end_time - timedelta(days=1)
            elif period == 'weekly':
                start_time = end_time - timedelta(weeks=1)
            elif period == 'monthly':
                start_time = end_time - timedelta(days=30)
            else:
                start_time = end_time - timedelta(days=1)
            
            # Get candlestick data for volatility calculation
            candlesticks = await self.db_connection.get_candlestick_data(
                symbol=symbol,
                timeframe='1h',
                start_time=start_time,
                limit=1000
            )
            
            if not candlesticks:
                return None
            
            # Calculate risk metrics
            prices = [c['close'] for c in candlesticks]
            returns = np.diff(np.log(prices))
            
            if len(returns) < 2:
                return None
            
            # Calculate VaR and CVaR
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            cvar_95 = np.mean(returns[returns <= var_95])
            cvar_99 = np.mean(returns[returns <= var_99])
            
            # Calculate volatility
            volatility = np.std(returns) * np.sqrt(252)
            
            # Calculate beta (simplified - using market correlation)
            beta = 1.0  # Placeholder for actual beta calculation
            correlation = 0.5  # Placeholder for actual correlation calculation
            
            # Calculate concentration and liquidity risk (simplified)
            concentration_risk = 0.1  # Placeholder
            liquidity_risk = 0.2  # Placeholder
            
            # Create risk report
            report = RiskReport(
                symbol=symbol,
                timestamp=datetime.now(),
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                volatility=volatility,
                beta=beta,
                correlation=correlation,
                concentration_risk=concentration_risk,
                liquidity_risk=liquidity_risk,
                metadata={
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'data_points': len(candlesticks),
                    'calculation_method': 'historical_simulation'
                }
            )
            
            # Store report
            self.risk_reports[symbol].append(report)
            if len(self.risk_reports[symbol]) > 50:
                self.risk_reports[symbol].popleft()
            
            # Update statistics
            self.stats['risk_reports'] += 1
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating risk report for {symbol}: {e}")
            return None
    
    async def _generate_market_report(self, symbol: str, period: str) -> Optional[MarketReport]:
        """Generate market report for a symbol"""
        try:
            # Get market regime from analytics engine
            if not self.analytics_engine:
                return None
            
            # Get current market regime
            symbol_analytics = self.analytics_engine.get_symbol_analytics(symbol)
            current_regime = symbol_analytics.get('market_regime')
            
            if not current_regime:
                return None
            
            # Get recent market data for trend analysis
            if not self.analytics_engine.real_time_pipeline:
                return None
            
            market_data = self.analytics_engine.real_time_pipeline.get_symbol_data(symbol, 'market_data')
            if len(market_data) < 20:
                return None
            
            # Analyze price and volume trends
            prices = [data.price for data in market_data[-20:]]
            volumes = [data.volume for data in market_data[-20:]]
            
            # Determine price trend
            if len(prices) >= 2:
                price_change = (prices[-1] - prices[0]) / prices[0]
                if price_change > 0.02:
                    price_trend = 'bullish'
                elif price_change < -0.02:
                    price_trend = 'bearish'
                else:
                    price_trend = 'sideways'
            else:
                price_trend = 'unknown'
            
            # Determine volume trend
            if len(volumes) >= 2:
                volume_change = (volumes[-1] - volumes[0]) / volumes[0]
                if volume_change > 0.1:
                    volume_trend = 'increasing'
                elif volume_change < -0.1:
                    volume_trend = 'decreasing'
                else:
                    volume_trend = 'stable'
            else:
                volume_trend = 'unknown'
            
            # Determine volatility regime
            if current_regime.characteristics.get('volatility', 0) > 0.8:
                volatility_regime = 'high'
            elif current_regime.characteristics.get('volatility', 0) > 0.4:
                volatility_regime = 'medium'
            else:
                volatility_regime = 'low'
            
            # Calculate support and resistance levels (simplified)
            if len(prices) >= 10:
                support_levels = [min(prices[-10:]) * 0.99, min(prices[-10:]) * 0.98]
                resistance_levels = [max(prices[-10:]) * 1.01, max(prices[-10:]) * 1.02]
            else:
                support_levels = []
                resistance_levels = []
            
            # Generate key insights
            key_insights = []
            if current_regime.confidence > 0.7:
                key_insights.append(f"Strong {current_regime.regime_type} regime detected")
            if price_trend != 'sideways':
                key_insights.append(f"Clear {price_trend} price trend")
            if volume_trend != 'stable':
                key_insights.append(f"Volume {volume_trend}")
            if volatility_regime == 'high':
                key_insights.append("High volatility - consider risk management")
            
            # Create market report
            report = MarketReport(
                symbol=symbol,
                timestamp=datetime.now(),
                market_regime=current_regime.regime_type,
                regime_confidence=current_regime.confidence,
                regime_duration=current_regime.duration,
                price_trend=price_trend,
                volume_trend=volume_trend,
                volatility_regime=volatility_regime,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                key_insights=key_insights,
                metadata={
                    'data_points': len(market_data),
                    'analysis_window': 20,
                    'regime_characteristics': current_regime.characteristics
                }
            )
            
            # Store report
            self.market_reports[symbol].append(report)
            if len(self.market_reports[symbol]) > 50:
                self.market_reports[symbol].popleft()
            
            # Update statistics
            self.stats['market_reports'] += 1
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating market report for {symbol}: {e}")
            return None
    
    async def _generate_report_summary(self, performance_report: Optional[PerformanceReport], 
                                     risk_report: Optional[RiskReport], 
                                     market_report: Optional[MarketReport]) -> Dict[str, Any]:
        """Generate summary of all reports"""
        try:
            summary = {
                'overall_score': 0.0,
                'risk_level': 'medium',
                'performance_rating': 'neutral',
                'market_outlook': 'neutral',
                'key_recommendations': []
            }
            
            # Calculate overall score
            score = 0.0
            
            # Performance contribution (40%)
            if performance_report:
                if performance_report.win_rate > 0.6:
                    score += 0.4
                    summary['performance_rating'] = 'excellent'
                elif performance_report.win_rate > 0.5:
                    score += 0.3
                    summary['performance_rating'] = 'good'
                elif performance_report.win_rate > 0.4:
                    score += 0.2
                    summary['performance_rating'] = 'fair'
                else:
                    score += 0.1
                    summary['performance_rating'] = 'poor'
            
            # Risk contribution (30%)
            if risk_report:
                if risk_report.volatility < 0.3:
                    score += 0.3
                    summary['risk_level'] = 'low'
                elif risk_report.volatility < 0.6:
                    score += 0.2
                    summary['risk_level'] = 'medium'
                else:
                    score += 0.1
                    summary['risk_level'] = 'high'
            
            # Market contribution (30%)
            if market_report:
                if market_report.regime_confidence > 0.8:
                    score += 0.3
                    summary['market_outlook'] = 'clear'
                elif market_report.regime_confidence > 0.6:
                    score += 0.2
                    summary['market_outlook'] = 'moderate'
                else:
                    score += 0.1
                    summary['market_outlook'] = 'unclear'
            
            summary['overall_score'] = min(1.0, score)
            
            # Generate key recommendations
            if performance_report and performance_report.win_rate < 0.5:
                summary['key_recommendations'].append("Consider reviewing trading strategy - low win rate")
            
            if risk_report and risk_report.volatility > 0.8:
                summary['key_recommendations'].append("High volatility detected - implement strict risk management")
            
            if market_report and market_report.regime_confidence < 0.5:
                summary['key_recommendations'].append("Unclear market regime - exercise caution with new positions")
            
            if not summary['key_recommendations']:
                summary['key_recommendations'].append("Market conditions appear favorable for current strategy")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating report summary: {e}")
            return {
                'overall_score': 0.0,
                'risk_level': 'unknown',
                'performance_rating': 'unknown',
                'market_outlook': 'unknown',
                'key_recommendations': ['Unable to generate summary']
            }
    
    # Public methods
    def add_callback(self, report_type: str, callback: Callable):
        """Add callback for report events"""
        self.report_callbacks[report_type].append(callback)
        self.logger.info(f"Added callback for {report_type} events")
    
    async def _trigger_callbacks(self, report_type: str, data: Any):
        """Trigger callbacks for report events"""
        callbacks = self.report_callbacks.get(report_type, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                self.logger.error(f"Error in {report_type} callback: {e}")
    
    def get_reporting_statistics(self) -> Dict[str, Any]:
        """Get reporting system statistics"""
        return {
            'stats': self.stats,
            'report_counts': {
                'performance': self.stats['performance_reports'],
                'risk': self.stats['risk_reports'],
                'market': self.stats['market_reports'],
                'combined': self.stats['total_reports_generated']
            },
            'symbols_with_reports': list(self.combined_reports.keys()),
            'last_report_time': self.stats['last_report'].isoformat() if self.stats['last_report'] else None
        }
    
    def get_symbol_reports(self, symbol: str, report_type: str = None) -> Dict[str, Any]:
        """Get reports for a specific symbol"""
        try:
            if report_type:
                if report_type == 'performance':
                    return {'performance': list(self.performance_reports.get(symbol, []))[-5:]}
                elif report_type == 'risk':
                    return {'risk': list(self.risk_reports.get(symbol, []))[-5:]}
                elif report_type == 'market':
                    return {'market': list(self.market_reports.get(symbol, []))[-5:]}
                else:
                    return {}
            else:
                return {
                    'performance': list(self.performance_reports.get(symbol, []))[-5:],
                    'risk': list(self.risk_reports.get(symbol, []))[-5:],
                    'market': list(self.market_reports.get(symbol, []))[-5:],
                    'combined': list(self.combined_reports.get(symbol, []))[-5:]
                }
        except Exception as e:
            self.logger.error(f"Error getting reports for {symbol}: {e}")
            return {}
    
    async def generate_all_symbols_report(self, period: str = 'daily') -> Dict[str, Any]:
        """Generate reports for all symbols"""
        try:
            all_reports = {}
            
            for symbol in self.symbols:
                report = await self.generate_comprehensive_report(symbol, period)
                if report:
                    all_reports[symbol] = report
            
            # Generate portfolio summary
            portfolio_summary = await self._generate_portfolio_summary(all_reports)
            
            return {
                'timestamp': datetime.now(),
                'period': period,
                'symbol_reports': all_reports,
                'portfolio_summary': portfolio_summary,
                'metadata': {
                    'total_symbols': len(self.symbols),
                    'reports_generated': len(all_reports),
                    'generation_time': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating all symbols report: {e}")
            return {}
    
    async def _generate_portfolio_summary(self, symbol_reports: Dict[str, Any]) -> Dict[str, Any]:
        """Generate portfolio-level summary"""
        try:
            if not symbol_reports:
                return {}
            
            # Aggregate performance metrics
            total_pnl = sum(r['performance']['total_pnl'] for r in symbol_reports.values() if r.get('performance'))
            total_trades = sum(r['performance']['total_trades'] for r in symbol_reports.values() if r.get('performance'))
            
            # Calculate portfolio risk metrics
            volatilities = [r['risk']['volatility'] for r in symbol_reports.values() if r.get('risk')]
            avg_volatility = np.mean(volatilities) if volatilities else 0
            
            # Determine portfolio outlook
            overall_scores = [r['summary']['overall_score'] for r in symbol_reports.values() if r.get('summary')]
            avg_score = np.mean(overall_scores) if overall_scores else 0
            
            if avg_score > 0.7:
                portfolio_outlook = 'bullish'
            elif avg_score > 0.4:
                portfolio_outlook = 'neutral'
            else:
                portfolio_outlook = 'bearish'
            
            return {
                'total_pnl': total_pnl,
                'total_trades': total_trades,
                'average_volatility': avg_volatility,
                'portfolio_score': avg_score,
                'portfolio_outlook': portfolio_outlook,
                'symbol_count': len(symbol_reports),
                'top_performers': sorted(
                    [(s, r['performance']['total_pnl']) for s, r in symbol_reports.items() if r.get('performance')],
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
            }
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio summary: {e}")
            return {}
    
    async def close(self):
        """Close the reporting system"""
        try:
            # Close components
            if self.analytics_engine:
                await self.analytics_engine.close()
            
            if self.db_connection:
                await self.db_connection.close()
            
            if self.trading_engine:
                await self.trading_engine.close()
            
            self.logger.info("Advanced Reporting System closed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to close reporting system: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
