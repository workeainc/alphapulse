"""
Performance Tracking System for AlphaPulse
Week 8: Strategy Configuration and Performance Monitoring

Features:
- Real-time performance metrics tracking
- Strategy performance analysis
- Risk monitoring and alerts
- Performance optimization insights

Author: AlphaPulse Team
Date: 2025
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for a single execution"""
    timestamp: datetime
    symbol: str
    strategy_id: str
    pnl: float
    win_rate: float
    drawdown: float
    execution_time: float
    trade_count: int
    avg_position_size: float
    max_position_size: float
    metadata: Dict[str, Any]

@dataclass
class PerformanceSummary:
    """Aggregated performance summary"""
    symbol: str
    strategy_id: str
    total_trades: int
    total_pnl: float
    avg_pnl: float
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    avg_execution_time: float
    avg_position_size: float
    max_position_size: float
    profit_factor: float
    max_consecutive_losses: int
    period_start: datetime
    period_end: datetime

class PerformanceTracker:
    """Comprehensive performance tracking system"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.logger = logger
        
        # Performance tracking
        self.metrics: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.alerts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Configuration
        self.alert_thresholds = {
            'max_daily_loss': -0.05,  # 5% daily loss
            'max_drawdown': -0.10,    # 10% drawdown
            'min_win_rate': 0.4,      # 40% win rate
            'max_execution_time': 0.1, # 100ms execution time
            'max_consecutive_losses': 5
        }
        
        # Performance optimization
        self.optimization_suggestions = []
        self.performance_trends = {}
    
    async def track_execution(self, symbol: str, strategy_id: str, execution: Dict[str, Any]) -> bool:
        """Track execution performance metrics"""
        try:
            # Calculate additional metrics
            pnl = execution.get('pnl', 0.0)
            execution_time = execution.get('execution_time', 0.0)
            position_size = execution.get('position_size', 0.0)
            
            # Calculate win rate (1 for profit, 0 for loss)
            win_rate = 1.0 if pnl > 0 else 0.0
            
            # Calculate drawdown (negative PnL)
            drawdown = max(0, -pnl)
            
            # Create metrics object
            metrics = PerformanceMetrics(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                strategy_id=strategy_id,
                pnl=pnl,
                win_rate=win_rate,
                drawdown=drawdown,
                execution_time=execution_time,
                trade_count=1,
                avg_position_size=position_size,
                max_position_size=position_size,
                metadata=execution.get('metadata', {})
            )
            
            # Store locally
            key = f"{symbol}_{strategy_id}"
            self.metrics[key].append(metrics)
            
            # Keep only last 1000 metrics per symbol/strategy
            if len(self.metrics[key]) > 1000:
                self.metrics[key] = self.metrics[key][-1000:]
            
            # Save to database
            success = await self.db.save_performance_metrics(asdict(metrics))
            
            if success:
                self.logger.info(f"Performance metrics tracked for {symbol} - {strategy_id}")
                
                # Check for alerts
                await self._check_alerts(symbol, strategy_id, metrics)
                
                # Update performance trends
                await self._update_performance_trends(symbol, strategy_id)
                
                # Generate optimization suggestions
                await self._generate_optimization_suggestions(symbol, strategy_id)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error tracking execution: {e}")
            return False
    
    async def _check_alerts(self, symbol: str, strategy_id: str, metrics: PerformanceMetrics) -> None:
        """Check for performance alerts"""
        try:
            key = f"{symbol}_{strategy_id}"
            
            # Get recent performance summary
            summary = await self.get_performance_summary(symbol, strategy_id, days=1)
            if not summary:
                return
            
            alerts = []
            
            # Check daily loss threshold
            if summary['total_pnl'] < self.alert_thresholds['max_daily_loss'] * 100:
                alerts.append({
                    'type': 'high_loss',
                    'severity': 'high',
                    'message': f"Daily loss {summary['total_pnl']:.2f}% exceeds threshold {self.alert_thresholds['max_daily_loss']*100:.1f}%",
                    'timestamp': datetime.now(timezone.utc)
                })
            
            # Check drawdown threshold
            if summary['max_drawdown'] > abs(self.alert_thresholds['max_drawdown'] * 100):
                alerts.append({
                    'type': 'high_drawdown',
                    'severity': 'medium',
                    'message': f"Drawdown {summary['max_drawdown']:.2f}% exceeds threshold {abs(self.alert_thresholds['max_drawdown']*100):.1f}%",
                    'timestamp': datetime.now(timezone.utc)
                })
            
            # Check win rate threshold
            if summary['win_rate'] < self.alert_thresholds['min_win_rate']:
                alerts.append({
                    'type': 'low_win_rate',
                    'severity': 'medium',
                    'message': f"Win rate {summary['win_rate']:.1%} below threshold {self.alert_thresholds['min_win_rate']:.1%}",
                    'timestamp': datetime.now(timezone.utc)
                })
            
            # Check execution time threshold
            if summary['avg_execution_time'] > self.alert_thresholds['max_execution_time']:
                alerts.append({
                    'type': 'slow_execution',
                    'severity': 'low',
                    'message': f"Execution time {summary['avg_execution_time']*1000:.1f}ms exceeds threshold {self.alert_thresholds['max_execution_time']*1000:.1f}ms",
                    'timestamp': datetime.now(timezone.utc)
                })
            
            # Store alerts
            if alerts:
                self.alerts[key].extend(alerts)
                self.logger.warning(f"Performance alerts generated for {symbol} - {strategy_id}: {len(alerts)} alerts")
                
                # Log high severity alerts
                for alert in alerts:
                    if alert['severity'] in ['high', 'medium']:
                        self.logger.error(f"ALERT: {alert['message']}")
            
        except Exception as e:
            self.logger.error(f"Error checking alerts: {e}")
    
    async def _update_performance_trends(self, symbol: str, strategy_id: str) -> None:
        """Update performance trends for optimization"""
        try:
            key = f"{symbol}_{strategy_id}"
            
            # Get performance data for trend analysis
            metrics_list = self.metrics.get(key, [])
            if len(metrics_list) < 10:
                return
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame([asdict(m) for m in metrics_list])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Calculate rolling metrics
            df['rolling_pnl'] = df['pnl'].rolling(window=10).mean()
            df['rolling_win_rate'] = df['win_rate'].rolling(window=10).mean()
            df['rolling_execution_time'] = df['execution_time'].rolling(window=10).mean()
            
            # Detect trends
            recent_pnl = df['rolling_pnl'].iloc[-5:].mean()
            earlier_pnl = df['rolling_pnl'].iloc[-15:-5].mean()
            
            recent_win_rate = df['rolling_win_rate'].iloc[-5:].mean()
            earlier_win_rate = df['rolling_win_rate'].iloc[-15:-5].mean()
            
            # Store trends
            self.performance_trends[key] = {
                'pnl_trend': 'improving' if recent_pnl > earlier_pnl else 'declining',
                'win_rate_trend': 'improving' if recent_win_rate > earlier_win_rate else 'declining',
                'recent_performance': {
                    'avg_pnl': recent_pnl,
                    'avg_win_rate': recent_win_rate,
                    'avg_execution_time': df['rolling_execution_time'].iloc[-5:].mean()
                },
                'last_updated': datetime.now(timezone.utc)
            }
            
        except Exception as e:
            self.logger.error(f"Error updating performance trends: {e}")
    
    async def _generate_optimization_suggestions(self, symbol: str, strategy_id: str) -> None:
        """Generate optimization suggestions based on performance"""
        try:
            key = f"{symbol}_{strategy_id}"
            trends = self.performance_trends.get(key, {})
            
            if not trends:
                return
            
            suggestions = []
            
            # PnL trend suggestions
            if trends['pnl_trend'] == 'declining':
                suggestions.append({
                    'type': 'pnl_optimization',
                    'priority': 'high',
                    'suggestion': f"PnL trend declining for {symbol} - {strategy_id}. Consider adjusting risk parameters or strategy logic.",
                    'timestamp': datetime.now(timezone.utc)
                })
            
            # Win rate suggestions
            if trends['win_rate_trend'] == 'declining':
                suggestions.append({
                    'type': 'win_rate_optimization',
                    'priority': 'medium',
                    'suggestion': f"Win rate declining for {symbol} - {strategy_id}. Review signal quality and entry criteria.",
                    'timestamp': datetime.now(timezone.utc)
                })
            
            # Execution time suggestions
            recent_exec_time = trends.get('recent_performance', {}).get('avg_execution_time', 0)
            if recent_exec_time > 0.05:  # 50ms
                suggestions.append({
                    'type': 'execution_optimization',
                    'priority': 'low',
                    'suggestion': f"Execution time high for {symbol} - {strategy_id}. Consider optimizing data processing pipeline.",
                    'timestamp': datetime.now(timezone.utc)
                })
            
            # Store suggestions
            if suggestions:
                self.optimization_suggestions.extend(suggestions)
                self.logger.info(f"Generated {len(suggestions)} optimization suggestions for {symbol} - {strategy_id}")
            
        except Exception as e:
            self.logger.error(f"Error generating optimization suggestions: {e}")
    
    async def get_performance_summary(self, symbol: str = None, strategy_id: str = None, 
                                     days: int = 30) -> Optional[Dict[str, Any]]:
        """Get performance summary from database"""
        try:
            # Try database first
            db_summary = await self.db.get_performance_summary(symbol, strategy_id, days)
            
            if db_summary:
                return db_summary
            
            # Fallback to local metrics
            return await self._calculate_local_summary(symbol, strategy_id, days)
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return None
    
    async def _calculate_local_summary(self, symbol: str = None, strategy_id: str = None, 
                                     days: int = 30) -> Optional[Dict[str, Any]]:
        """Calculate performance summary from local metrics"""
        try:
            # Filter metrics by symbol and strategy
            filtered_metrics = []
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
            
            for key, metrics_list in self.metrics.items():
                if symbol and strategy_id:
                    if key == f"{symbol}_{strategy_id}":
                        filtered_metrics.extend(metrics_list)
                elif symbol:
                    if key.startswith(f"{symbol}_"):
                        filtered_metrics.extend(metrics_list)
                elif strategy_id:
                    if key.endswith(f"_{strategy_id}"):
                        filtered_metrics.extend(metrics_list)
                else:
                    filtered_metrics.extend(metrics_list)
            
            # Filter by time
            filtered_metrics = [m for m in filtered_metrics if m.timestamp >= cutoff_time]
            
            if not filtered_metrics:
                return None
            
            # Calculate summary statistics
            pnl_values = [m.pnl for m in filtered_metrics]
            win_rates = [m.win_rate for m in filtered_metrics]
            execution_times = [m.execution_time for m in filtered_metrics]
            position_sizes = [m.avg_position_size for m in filtered_metrics]
            
            # Calculate drawdown
            cumulative_pnl = np.cumsum(pnl_values)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdowns = running_max - cumulative_pnl
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
            
            # Calculate Sharpe ratio (simplified)
            avg_pnl = np.mean(pnl_values) if pnl_values else 0
            std_pnl = np.std(pnl_values) if len(pnl_values) > 1 else 0
            sharpe_ratio = avg_pnl / std_pnl if std_pnl > 0 else 0
            
            # Calculate profit factor
            profits = [p for p in pnl_values if p > 0]
            losses = [abs(p) for p in pnl_values if p < 0]
            profit_factor = sum(profits) / sum(losses) if sum(losses) > 0 else float('inf')
            
            # Calculate consecutive losses
            consecutive_losses = 0
            max_consecutive_losses = 0
            for pnl in pnl_values:
                if pnl < 0:
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                else:
                    consecutive_losses = 0
            
            return {
                'total_trades': len(filtered_metrics),
                'total_pnl': sum(pnl_values),
                'avg_pnl': avg_pnl,
                'win_rate': np.mean(win_rates) if win_rates else 0,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'avg_execution_time': np.mean(execution_times) if execution_times else 0,
                'avg_position_size': np.mean(position_sizes) if position_sizes else 0,
                'max_position_size': max(position_sizes) if position_sizes else 0,
                'profit_factor': profit_factor,
                'max_consecutive_losses': max_consecutive_losses,
                'period_start': min(m.timestamp for m in filtered_metrics),
                'period_end': max(m.timestamp for m in filtered_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating local summary: {e}")
            return None
    
    async def get_alerts(self, symbol: str = None, strategy_id: str = None, 
                        severity: str = None) -> List[Dict[str, Any]]:
        """Get performance alerts"""
        try:
            all_alerts = []
            
            for key, alerts in self.alerts.items():
                if symbol and strategy_id:
                    if key == f"{symbol}_{strategy_id}":
                        all_alerts.extend(alerts)
                elif symbol:
                    if key.startswith(f"{symbol}_"):
                        all_alerts.extend(alerts)
                elif strategy_id:
                    if key.endswith(f"_{strategy_id}"):
                        all_alerts.extend(alerts)
                else:
                    all_alerts.extend(alerts)
            
            # Filter by severity if specified
            if severity:
                all_alerts = [a for a in all_alerts if a['severity'] == severity]
            
            # Sort by timestamp (newest first)
            all_alerts.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return all_alerts
            
        except Exception as e:
            self.logger.error(f"Error getting alerts: {e}")
            return []
    
    async def get_optimization_suggestions(self, symbol: str = None, 
                                         strategy_id: str = None) -> List[Dict[str, Any]]:
        """Get optimization suggestions"""
        try:
            suggestions = self.optimization_suggestions.copy()
            
            # Filter by symbol and strategy if specified
            if symbol or strategy_id:
                filtered_suggestions = []
                for suggestion in suggestions:
                    if symbol and strategy_id:
                        if f"{symbol} - {strategy_id}" in suggestion['suggestion']:
                            filtered_suggestions.append(suggestion)
                    elif symbol:
                        if symbol in suggestion['suggestion']:
                            filtered_suggestions.append(suggestion)
                    elif strategy_id:
                        if strategy_id in suggestion['suggestion']:
                            filtered_suggestions.append(suggestion)
                suggestions = filtered_suggestions
            
            # Sort by priority and timestamp
            priority_order = {'high': 3, 'medium': 2, 'low': 1}
            suggestions.sort(key=lambda x: (priority_order.get(x['priority'], 0), x['timestamp']), reverse=True)
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error getting optimization suggestions: {e}")
            return []
    
    async def get_performance_trends(self, symbol: str = None, 
                                   strategy_id: str = None) -> Dict[str, Any]:
        """Get performance trends"""
        try:
            if symbol and strategy_id:
                key = f"{symbol}_{strategy_id}"
                return self.performance_trends.get(key, {})
            elif symbol:
                return {k: v for k, v in self.performance_trends.items() if k.startswith(f"{symbol}_")}
            elif strategy_id:
                return {k: v for k, v in self.performance_trends.items() if k.endswith(f"_{strategy_id}")}
            else:
                return self.performance_trends
                
        except Exception as e:
            self.logger.error(f"Error getting performance trends: {e}")
            return {}
    
    async def clear_old_data(self, days: int = 90) -> None:
        """Clear old performance data"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
            
            # Clear old metrics
            for key in list(self.metrics.keys()):
                self.metrics[key] = [m for m in self.metrics[key] if m.timestamp >= cutoff_time]
                if not self.metrics[key]:
                    del self.metrics[key]
            
            # Clear old alerts
            for key in list(self.alerts.keys()):
                self.alerts[key] = [a for a in self.alerts[key] if a['timestamp'] >= cutoff_time]
                if not self.alerts[key]:
                    del self.alerts[key]
            
            # Clear old optimization suggestions
            self.optimization_suggestions = [s for s in self.optimization_suggestions 
                                          if s['timestamp'] >= cutoff_time]
            
            self.logger.info(f"Cleared performance data older than {days} days")
            
        except Exception as e:
            self.logger.error(f"Error clearing old data: {e}")
    
    async def close(self):
        """Close the performance tracker"""
        try:
            self.metrics.clear()
            self.alerts.clear()
            self.optimization_suggestions.clear()
            self.performance_trends.clear()
            self.logger.info("Performance tracker closed")
        except Exception as e:
            self.logger.error(f"Error closing performance tracker: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
