"""
Real-time Performance Monitoring Dashboard for AlphaPlus
Monitors trading performance, system health, and generates insights
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
import json
import os
from dataclasses import dataclass, asdict

# Import our enhanced components
try:
    from ..database.connection import TimescaleDBConnection
    from ..core.trading_engine import TradingEngine
    from ..strategies.strategy_manager import StrategyManager
    from ..execution.order_manager import OrderManager
except ImportError:
    # Fallback for testing
    TimescaleDBConnection = None
    TradingEngine = None
    StrategyManager = None
    OrderManager = None

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Real-time performance metrics"""
    timestamp: datetime
    total_pnl: float
    daily_pnl: float
    win_rate: float
    total_trades: int
    active_positions: int
    portfolio_value: float
    risk_metrics: Dict[str, float]
    system_health: Dict[str, Any]

class PerformanceDashboard:
    """Real-time performance monitoring dashboard"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Dashboard configuration
        self.update_interval = self.config.get('update_interval', 30)  # seconds
        self.history_length = self.config.get('history_length', 1000)
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'max_daily_loss': -0.05,  # -5%
            'max_drawdown': -0.15,     # -15%
            'min_win_rate': 0.4,       # 40%
            'max_position_risk': 0.1   # 10%
        })
        
        # Component references
        self.db_connection = None
        self.trading_engine = None
        self.strategy_manager = None
        self.order_manager = None
        
        # Performance tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.alerts: List[Dict[str, Any]] = []
        self.monitoring_task = None
        self.is_monitoring = False
        
        # Dashboard data
        self.current_metrics = None
        self.performance_summary = {}
        self.risk_analysis = {}
        self.system_status = {}
        
    async def initialize(self):
        """Initialize the performance dashboard"""
        try:
            self.logger.info("Initializing Performance Dashboard...")
            
            # Initialize database connection
            if TimescaleDBConnection:
                self.db_connection = TimescaleDBConnection(
                    self.config.get('db_config', {})
                )
                await self.db_connection.initialize()
            
            # Initialize trading components (if available)
            if TradingEngine:
                self.trading_engine = TradingEngine()
                await self.trading_engine.initialize()
            
            if StrategyManager:
                self.strategy_manager = StrategyManager()
                await self.strategy_manager.initialize()
            
            if OrderManager:
                self.order_manager = OrderManager()
                await self.order_manager.initialize()
            
            self.logger.info("Performance Dashboard initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Performance Dashboard: {e}")
            raise
    
    async def start_monitoring(self):
        """Start real-time performance monitoring"""
        try:
            if self.is_monitoring:
                self.logger.warning("Monitoring already active")
                return
            
            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            self.logger.info("Performance monitoring started")
            
        except Exception as e:
            self.logger.error(f"Error starting monitoring: {e}")
            raise
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        try:
            self.is_monitoring = False
            
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("Performance monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring: {e}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while self.is_monitoring:
                # Collect performance metrics
                metrics = await self._collect_performance_metrics()
                
                # Update dashboard data
                await self._update_dashboard(metrics)
                
                # Check for alerts
                await self._check_alerts(metrics)
                
                # Store metrics in history
                self._store_metrics(metrics)
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
        except asyncio.CancelledError:
            self.logger.info("Monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {e}")
            self.is_monitoring = False
    
    async def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect real-time performance metrics"""
        try:
            timestamp = datetime.now(timezone.utc)
            
            # Get trading performance
            trading_performance = await self._get_trading_performance()
            
            # Get portfolio metrics
            portfolio_metrics = await self._get_portfolio_metrics()
            
            # Get risk metrics
            risk_metrics = await self._get_risk_metrics()
            
            # Get system health
            system_health = await self._get_system_health()
            
            # Create performance metrics
            metrics = PerformanceMetrics(
                timestamp=timestamp,
                total_pnl=trading_performance.get('total_pnl', 0.0),
                daily_pnl=trading_performance.get('daily_pnl', 0.0),
                win_rate=trading_performance.get('win_rate', 0.0),
                total_trades=trading_performance.get('total_trades', 0),
                active_positions=trading_performance.get('active_positions', 0),
                portfolio_value=portfolio_metrics.get('total_value', 0.0),
                risk_metrics=risk_metrics,
                system_health=system_health
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting performance metrics: {e}")
            # Return default metrics
            return PerformanceMetrics(
                timestamp=datetime.now(timezone.utc),
                total_pnl=0.0,
                daily_pnl=0.0,
                win_rate=0.0,
                total_trades=0,
                active_positions=0,
                portfolio_value=0.0,
                risk_metrics={},
                system_health={'status': 'error', 'error': str(e)}
            )
    
    async def _get_trading_performance(self) -> Dict[str, Any]:
        """Get trading performance metrics"""
        try:
            if self.trading_engine:
                return await self.trading_engine.get_performance_summary()
            elif self.db_connection:
                return await self.db_connection.get_performance_summary(days=1)
            else:
                return {
                    'total_pnl': 0.0,
                    'daily_pnl': 0.0,
                    'win_rate': 0.0,
                    'total_trades': 0,
                    'active_positions': 0
                }
                
        except Exception as e:
            self.logger.error(f"Error getting trading performance: {e}")
            return {}
    
    async def _get_portfolio_metrics(self) -> Dict[str, Any]:
        """Get portfolio metrics"""
        try:
            if self.trading_engine:
                # Get portfolio value from trading engine
                return {'total_value': 10000.0}  # Placeholder
            else:
                return {'total_value': 10000.0}  # Default value
                
        except Exception as e:
            self.logger.error(f"Error getting portfolio metrics: {e}")
            return {'total_value': 0.0}
    
    async def _get_risk_metrics(self) -> Dict[str, float]:
        """Get risk metrics"""
        try:
            risk_metrics = {}
            
            # Calculate drawdown
            if len(self.performance_history) > 0:
                peak_value = max([m.portfolio_value for m in self.performance_history])
                current_value = self.performance_history[-1].portfolio_value if self.performance_history else 0
                if peak_value > 0:
                    risk_metrics['drawdown'] = (current_value - peak_value) / peak_value
                else:
                    risk_metrics['drawdown'] = 0.0
            else:
                risk_metrics['drawdown'] = 0.0
            
            # Calculate volatility (simplified)
            if len(self.performance_history) > 10:
                returns = []
                for i in range(1, len(self.performance_history)):
                    prev_value = self.performance_history[i-1].portfolio_value
                    curr_value = self.performance_history[i].portfolio_value
                    if prev_value > 0:
                        returns.append((curr_value - prev_value) / prev_value)
                
                if returns:
                    risk_metrics['volatility'] = sum(returns) / len(returns)
                else:
                    risk_metrics['volatility'] = 0.0
            else:
                risk_metrics['volatility'] = 0.0
            
            # Position risk
            if self.trading_engine:
                risk_metrics['position_risk'] = 0.05  # Placeholder
            else:
                risk_metrics['position_risk'] = 0.0
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting risk metrics: {e}")
            return {}
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get system health status"""
        try:
            system_health = {
                'overall_status': 'healthy',
                'components': {},
                'warnings': [],
                'errors': []
            }
            
            # Check trading engine health
            if self.trading_engine:
                try:
                    engine_health = await self.trading_engine.health_check()
                    system_health['components']['trading_engine'] = engine_health
                    
                    if engine_health.get('status') != 'healthy':
                        system_health['warnings'].append('Trading engine issues detected')
                        system_health['overall_status'] = 'degraded'
                except Exception as e:
                    system_health['components']['trading_engine'] = {'status': 'error', 'error': str(e)}
                    system_health['errors'].append(f'Trading engine error: {e}')
                    system_health['overall_status'] = 'unhealthy'
            
            # Check strategy manager health
            if self.strategy_manager:
                try:
                    strategy_health = await self.strategy_manager.health_check()
                    system_health['components']['strategy_manager'] = strategy_health
                    
                    if strategy_health.get('status') != 'healthy':
                        system_health['warnings'].append('Strategy manager issues detected')
                        system_health['overall_status'] = 'degraded'
                except Exception as e:
                    system_health['components']['strategy_manager'] = {'status': 'error', 'error': str(e)}
                    system_health['errors'].append(f'Strategy manager error: {e}')
                    system_health['overall_status'] = 'unhealthy'
            
            # Check order manager health
            if self.order_manager:
                try:
                    order_health = await self.order_manager.health_check()
                    system_health['components']['order_manager'] = order_health
                    
                    if order_health.get('status') != 'healthy':
                        system_health['warnings'].append('Order manager issues detected')
                        system_health['overall_status'] = 'degraded'
                except Exception as e:
                    system_health['components']['order_manager'] = {'status': 'error', 'error': str(e)}
                    system_health['errors'].append(f'Order manager error: {e}')
                    system_health['overall_status'] = 'unhealthy'
            
            # Check database health
            if self.db_connection:
                try:
                    db_health = await self.db_connection.health_check()
                    system_health['components']['database'] = db_health
                    
                    if db_health.get('status') != 'healthy':
                        system_health['warnings'].append('Database connection issues detected')
                        system_health['overall_status'] = 'degraded'
                except Exception as e:
                    system_health['components']['database'] = {'status': 'error', 'error': str(e)}
                    system_health['errors'].append(f'Database error: {e}')
                    system_health['overall_status'] = 'unhealthy'
            
            return system_health
            
        except Exception as e:
            self.logger.error(f"Error getting system health: {e}")
            return {
                'overall_status': 'error',
                'error': str(e),
                'components': {},
                'warnings': [],
                'errors': [str(e)]
            }
    
    async def _update_dashboard(self, metrics: PerformanceMetrics):
        """Update dashboard data with new metrics"""
        try:
            # Update current metrics
            self.current_metrics = metrics
            
            # Update performance summary
            self.performance_summary = {
                'total_pnl': metrics.total_pnl,
                'daily_pnl': metrics.daily_pnl,
                'win_rate': metrics.win_rate,
                'total_trades': metrics.total_trades,
                'active_positions': metrics.active_positions,
                'portfolio_value': metrics.portfolio_value,
                'last_update': metrics.timestamp.isoformat()
            }
            
            # Update risk analysis
            self.risk_analysis = {
                'drawdown': metrics.risk_metrics.get('drawdown', 0.0),
                'volatility': metrics.risk_metrics.get('volatility', 0.0),
                'position_risk': metrics.risk_metrics.get('position_risk', 0.0),
                'risk_level': self._calculate_risk_level(metrics.risk_metrics)
            }
            
            # Update system status
            self.system_status = {
                'overall_status': metrics.system_health.get('overall_status', 'unknown'),
                'component_count': len(metrics.system_health.get('components', {})),
                'warning_count': len(metrics.system_health.get('warnings', [])),
                'error_count': len(metrics.system_health.get('errors', [])),
                'last_check': metrics.timestamp.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating dashboard: {e}")
    
    def _calculate_risk_level(self, risk_metrics: Dict[str, float]) -> str:
        """Calculate overall risk level"""
        try:
            risk_score = 0
            
            # Drawdown risk
            drawdown = abs(risk_metrics.get('drawdown', 0))
            if drawdown > 0.2:  # >20%
                risk_score += 3
            elif drawdown > 0.1:  # >10%
                risk_score += 2
            elif drawdown > 0.05:  # >5%
                risk_score += 1
            
            # Volatility risk
            volatility = abs(risk_metrics.get('volatility', 0))
            if volatility > 0.05:  # >5%
                risk_score += 2
            elif volatility > 0.02:  # >2%
                risk_score += 1
            
            # Position risk
            position_risk = risk_metrics.get('position_risk', 0)
            if position_risk > 0.1:  # >10%
                risk_score += 2
            elif position_risk > 0.05:  # >5%
                risk_score += 1
            
            # Determine risk level
            if risk_score >= 5:
                return 'high'
            elif risk_score >= 3:
                return 'medium'
            elif risk_score >= 1:
                return 'low'
            else:
                return 'minimal'
                
        except Exception as e:
            self.logger.error(f"Error calculating risk level: {e}")
            return 'unknown'
    
    async def _check_alerts(self, metrics: PerformanceMetrics):
        """Check for performance alerts"""
        try:
            alerts = []
            
            # Daily loss alert
            if metrics.daily_pnl < self.alert_thresholds['max_daily_loss']:
                alerts.append({
                    'type': 'daily_loss',
                    'severity': 'high',
                    'message': f'Daily loss threshold exceeded: {metrics.daily_pnl:.2%}',
                    'timestamp': metrics.timestamp.isoformat(),
                    'value': metrics.daily_pnl,
                    'threshold': self.alert_thresholds['max_daily_loss']
                })
            
            # Drawdown alert
            drawdown = metrics.risk_metrics.get('drawdown', 0)
            if drawdown < self.alert_thresholds['max_drawdown']:
                alerts.append({
                    'type': 'drawdown',
                    'severity': 'high',
                    'message': f'Maximum drawdown threshold exceeded: {drawdown:.2%}',
                    'timestamp': metrics.timestamp.isoformat(),
                    'value': drawdown,
                    'threshold': self.alert_thresholds['max_drawdown']
                })
            
            # Win rate alert
            if metrics.win_rate < self.alert_thresholds['min_win_rate']:
                alerts.append({
                    'type': 'win_rate',
                    'severity': 'medium',
                    'message': f'Win rate below threshold: {metrics.win_rate:.2%}',
                    'timestamp': metrics.timestamp.isoformat(),
                    'value': metrics.win_rate,
                    'threshold': self.alert_thresholds['min_win_rate']
                })
            
            # Position risk alert
            position_risk = metrics.risk_metrics.get('position_risk', 0)
            if position_risk > self.alert_thresholds['max_position_risk']:
                alerts.append({
                    'type': 'position_risk',
                    'severity': 'medium',
                    'message': f'Position risk above threshold: {position_risk:.2%}',
                    'timestamp': metrics.timestamp.isoformat(),
                    'value': position_risk,
                    'threshold': self.alert_thresholds['max_position_risk']
                })
            
            # System health alerts
            if metrics.system_health.get('overall_status') == 'unhealthy':
                alerts.append({
                    'type': 'system_health',
                    'severity': 'critical',
                    'message': 'System health critical - immediate attention required',
                    'timestamp': metrics.timestamp.isoformat(),
                    'errors': metrics.system_health.get('errors', [])
                })
            
            # Add new alerts
            self.alerts.extend(alerts)
            
            # Keep only recent alerts (last 100)
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]
            
            # Log high severity alerts
            for alert in alerts:
                if alert['severity'] in ['high', 'critical']:
                    self.logger.warning(f"ALERT: {alert['message']}")
                    
        except Exception as e:
            self.logger.error(f"Error checking alerts: {e}")
    
    def _store_metrics(self, metrics: PerformanceMetrics):
        """Store metrics in history"""
        try:
            self.performance_history.append(metrics)
            
            # Keep only recent history
            if len(self.performance_history) > self.history_length:
                self.performance_history = self.performance_history[-self.history_length:]
                
        except Exception as e:
            self.logger.error(f"Error storing metrics: {e}")
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        try:
            return {
                'performance_summary': self.performance_summary,
                'risk_analysis': self.risk_analysis,
                'system_status': self.system_status,
                'recent_alerts': self.alerts[-10:],  # Last 10 alerts
                'performance_trend': self._get_performance_trend(),
                'last_update': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting dashboard data: {e}")
            return {'error': str(e)}
    
    def _get_performance_trend(self) -> List[Dict[str, Any]]:
        """Get performance trend data"""
        try:
            if len(self.performance_history) < 2:
                return []
            
            # Get last 50 data points for trend
            recent_history = self.performance_history[-50:]
            
            trend_data = []
            for metrics in recent_history:
                trend_data.append({
                    'timestamp': metrics.timestamp.isoformat(),
                    'portfolio_value': metrics.portfolio_value,
                    'daily_pnl': metrics.daily_pnl,
                    'win_rate': metrics.win_rate,
                    'active_positions': metrics.active_positions
                })
            
            return trend_data
            
        except Exception as e:
            self.logger.error(f"Error getting performance trend: {e}")
            return []
    
    async def export_dashboard_report(self, format: str = 'json') -> str:
        """Export dashboard report"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format == 'json':
                filename = f"dashboard_report_{timestamp}.json"
                
                report_data = {
                    'performance_summary': self.performance_summary,
                    'risk_analysis': self.risk_analysis,
                    'system_status': self.system_status,
                    'alerts': self.alerts,
                    'performance_history': [asdict(m) for m in self.performance_history[-100:]],  # Last 100 metrics
                    'export_timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                with open(filename, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
                
                return filename
            
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            self.logger.error(f"Error exporting dashboard report: {e}")
            return f"Error exporting report: {e}"
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for performance dashboard"""
        try:
            return {
                'status': 'healthy' if self.is_monitoring else 'stopped',
                'monitoring_active': self.is_monitoring,
                'update_interval': self.update_interval,
                'history_length': len(self.performance_history),
                'alert_count': len(self.alerts),
                'last_metrics': self.current_metrics.timestamp.isoformat() if self.current_metrics else None
            }
            
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def close(self):
        """Close performance dashboard"""
        try:
            await self.stop_monitoring()
            
            if self.db_connection:
                await self.db_connection.close()
            
            if self.trading_engine:
                # Note: TradingEngine doesn't have a close method yet
                pass
            
            self.logger.info("Performance dashboard closed")
            
        except Exception as e:
            self.logger.error(f"Error closing performance dashboard: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
