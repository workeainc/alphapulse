"""
Performance Analyzer for Outcome Tracking

Analyzes trading performance metrics and generates insights
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import statistics

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of performance metrics"""
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    AVG_WIN = "avg_win"
    AVG_LOSS = "avg_loss"
    TOTAL_TRADES = "total_trades"
    PROFITABLE_TRADES = "profitable_trades"
    LOSING_TRADES = "losing_trades"

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_trades: int = 0
    profitable_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_trade_duration: timedelta = field(default_factory=lambda: timedelta())
    risk_reward_ratio: float = 0.0
    expectancy: float = 0.0

@dataclass
class PerformanceInsight:
    """Performance insight data structure"""
    insight_type: str
    description: str
    severity: str  # 'info', 'warning', 'critical'
    recommendation: str
    metrics_affected: List[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class PerformanceAnalyzer:
    """Analyzes trading performance and generates insights"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.is_initialized = False
        self.is_running = False
        self.analysis_interval = self.config.get('analysis_interval', 300)  # 5 minutes
        self.min_trades_for_analysis = self.config.get('min_trades_for_analysis', 10)
        self.analysis_task = None
        self.performance_history: List[PerformanceMetrics] = []
        self.insights_history: List[PerformanceInsight] = []
        
        # Fallback classes for import errors
        try:
            from database.connection import TimescaleDBConnection
            from core.config import settings
        except ImportError:
            logger.warning("Using fallback classes for PerformanceAnalyzer")
            
            @dataclass
            class TimescaleDBConnection:
                def __init__(self, config=None):
                    self.config = config or {}
                    self.is_initialized = False
                async def initialize(self): pass
                async def shutdown(self): pass
                async def close(self):
                    self.is_initialized = False
            
            @dataclass
            class settings:
                TIMESCALEDB_HOST = 'localhost'
                TIMESCALEDB_PORT = 5432
                TIMESCALEDB_DATABASE = 'alphapulse'
                TIMESCALEDB_USERNAME = 'alpha_emon'
                TIMESCALEDB_PASSWORD = 'Emon_@17711'
    
    async def initialize(self):
        """Initialize the performance analyzer"""
        try:
            logger.info("🔧 Initializing Performance Analyzer...")
            
            # Initialize database connection if needed
            if self.config.get('enable_database_storage', True):
                from database.connection import TimescaleDBConnection
                from core.config import settings
                
                self.db_connection = TimescaleDBConnection({
                    'host': settings.TIMESCALEDB_HOST,
                    'port': settings.TIMESCALEDB_PORT,
                    'database': settings.TIMESCALEDB_DATABASE,
                    'username': settings.TIMESCALEDB_USERNAME,
                    'password': settings.TIMESCALEDB_PASSWORD
                })
                await self.db_connection.initialize()
            
            self.is_initialized = True
            logger.info("✅ Performance Analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Performance Analyzer: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the performance analyzer"""
        try:
            logger.info("🛑 Shutting down Performance Analyzer...")
            
            self.is_running = False
            
            if hasattr(self, 'analysis_task') and self.analysis_task:
                self.analysis_task.cancel()
                try:
                    await self.analysis_task
                except asyncio.CancelledError:
                    pass
            
            if hasattr(self, 'db_connection'):
                await self.db_connection.close()
            
            logger.info("✅ Performance Analyzer shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Error during Performance Analyzer shutdown: {e}")
    
    async def start_analysis(self):
        """Start continuous performance analysis"""
        if not self.is_initialized:
            raise RuntimeError("Performance Analyzer not initialized")
        
        self.is_running = True
        self.analysis_task = asyncio.create_task(self._analysis_loop())
        logger.info("🚀 Performance analysis started")
    
    async def stop_analysis(self):
        """Stop continuous performance analysis"""
        self.is_running = False
        if self.analysis_task:
            self.analysis_task.cancel()
            try:
                await self.analysis_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Performance analysis stopped")
    
    async def _analysis_loop(self):
        """Main analysis loop"""
        while self.is_running:
            try:
                await self.analyze_performance()
                await asyncio.sleep(self.analysis_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in analysis loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def analyze_performance(self, time_period: Optional[timedelta] = None) -> PerformanceMetrics:
        """Analyze performance for a given time period"""
        try:
            # Get outcomes from database
            outcomes = await self._get_outcomes(time_period)
            
            if len(outcomes) < self.min_trades_for_analysis:
                logger.info(f"Not enough trades for analysis ({len(outcomes)} < {self.min_trades_for_analysis})")
                return PerformanceMetrics()
            
            # Calculate metrics
            metrics = self._calculate_metrics(outcomes)
            
            # Generate insights
            insights = self._generate_insights(metrics, outcomes)
            
            # Store results
            self.performance_history.append(metrics)
            self.insights_history.extend(insights)
            
            # Store in database if enabled
            if hasattr(self, 'db_connection'):
                await self._store_analysis_results(metrics, insights)
            
            logger.info(f"📊 Performance analysis completed: {metrics.win_rate:.2%} win rate, {metrics.total_pnl:.2f} total PnL")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Performance analysis failed: {e}")
            return PerformanceMetrics()
    
    async def _get_outcomes(self, time_period: Optional[timedelta] = None) -> List[Dict[str, Any]]:
        """Get outcomes from database"""
        try:
            if not hasattr(self, 'db_connection'):
                return []
            
            async with self.db_connection.get_session() as session:
                query = """
                    SELECT 
                        signal_id, outcome_type, exit_price, exit_timestamp,
                        realized_pnl, time_to_exit, order_type, order_state
                    FROM signal_outcomes 
                    WHERE order_state = 'filled'
                """
                
                if time_period:
                    cutoff_time = datetime.now(timezone.utc) - time_period
                    query += " AND exit_timestamp >= :cutoff_time"
                    result = await session.execute(query, {'cutoff_time': cutoff_time})
                else:
                    result = await session.execute(query)
                
                outcomes = []
                for row in result.fetchall():
                    outcomes.append({
                        'signal_id': row[0],
                        'outcome_type': row[1],
                        'exit_price': row[2],
                        'exit_timestamp': row[3],
                        'realized_pnl': row[4],
                        'time_to_exit': row[5],
                        'order_type': row[6],
                        'order_state': row[7]
                    })
                
                return outcomes
                
        except Exception as e:
            logger.error(f"❌ Failed to get outcomes: {e}")
            return []
    
    def _calculate_metrics(self, outcomes: List[Dict[str, Any]]) -> PerformanceMetrics:
        """Calculate performance metrics from outcomes"""
        if not outcomes:
            return PerformanceMetrics()
        
        # Extract PnL values
        pnl_values = [outcome['realized_pnl'] for outcome in outcomes]
        profitable_trades = [pnl for pnl in pnl_values if pnl > 0]
        losing_trades = [pnl for pnl in pnl_values if pnl < 0]
        
        # Basic metrics
        total_trades = len(outcomes)
        profitable_count = len(profitable_trades)
        losing_count = len(losing_trades)
        total_pnl = sum(pnl_values)
        
        # Win rate
        win_rate = profitable_count / total_trades if total_trades > 0 else 0.0
        
        # Average win/loss
        avg_win = statistics.mean(profitable_trades) if profitable_trades else 0.0
        avg_loss = abs(statistics.mean(losing_trades)) if losing_trades else 0.0
        
        # Profit factor
        total_wins = sum(profitable_trades) if profitable_trades else 0.0
        total_losses = abs(sum(losing_trades)) if losing_trades else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Risk/reward ratio
        risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
        
        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Max consecutive wins/losses
        max_consecutive_wins = self._calculate_max_consecutive(pnl_values, positive=True)
        max_consecutive_losses = self._calculate_max_consecutive(pnl_values, positive=False)
        
        # Max drawdown
        max_drawdown = self._calculate_max_drawdown(pnl_values)
        
        # Sharpe ratio (simplified)
        sharpe_ratio = self._calculate_sharpe_ratio(pnl_values)
        
        # Average trade duration
        durations = [outcome['time_to_exit'] for outcome in outcomes if outcome['time_to_exit']]
        avg_duration = statistics.mean(durations) if durations else timedelta()
        
        return PerformanceMetrics(
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=total_trades,
            profitable_trades=profitable_count,
            losing_trades=losing_count,
            total_pnl=total_pnl,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            avg_trade_duration=avg_duration,
            risk_reward_ratio=risk_reward_ratio,
            expectancy=expectancy
        )
    
    def _calculate_max_consecutive(self, pnl_values: List[float], positive: bool = True) -> int:
        """Calculate max consecutive wins or losses"""
        max_consecutive = 0
        current_consecutive = 0
        
        for pnl in pnl_values:
            if (positive and pnl > 0) or (not positive and pnl < 0):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_max_drawdown(self, pnl_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not pnl_values:
            return 0.0
        
        cumulative = []
        running_total = 0
        for pnl in pnl_values:
            running_total += pnl
            cumulative.append(running_total)
        
        max_drawdown = 0
        peak = cumulative[0]
        
        for value in cumulative:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_sharpe_ratio(self, pnl_values: List[float]) -> float:
        """Calculate Sharpe ratio (simplified)"""
        if len(pnl_values) < 2:
            return 0.0
        
        mean_return = statistics.mean(pnl_values)
        std_return = statistics.stdev(pnl_values)
        
        if std_return == 0:
            return 0.0
        
        # Assuming risk-free rate of 0 for simplicity
        return mean_return / std_return
    
    def _generate_insights(self, metrics: PerformanceMetrics, outcomes: List[Dict[str, Any]]) -> List[PerformanceInsight]:
        """Generate performance insights"""
        insights = []
        
        # Win rate insights
        if metrics.win_rate < 0.4:
            insights.append(PerformanceInsight(
                insight_type="low_win_rate",
                description=f"Win rate is low ({metrics.win_rate:.1%})",
                severity="warning",
                recommendation="Consider improving entry criteria or risk management",
                metrics_affected=["win_rate"]
            ))
        elif metrics.win_rate > 0.7:
            insights.append(PerformanceInsight(
                insight_type="high_win_rate",
                description=f"Excellent win rate ({metrics.win_rate:.1%})",
                severity="info",
                recommendation="Strategy is performing well, consider increasing position sizes",
                metrics_affected=["win_rate"]
            ))
        
        # Profit factor insights
        if metrics.profit_factor < 1.2:
            insights.append(PerformanceInsight(
                insight_type="low_profit_factor",
                description=f"Profit factor is low ({metrics.profit_factor:.2f})",
                severity="warning",
                recommendation="Focus on improving risk/reward ratio",
                metrics_affected=["profit_factor"]
            ))
        
        # Drawdown insights
        if metrics.max_drawdown > 0.2:
            insights.append(PerformanceInsight(
                insight_type="high_drawdown",
                description=f"High maximum drawdown ({metrics.max_drawdown:.1%})",
                severity="critical",
                recommendation="Implement stricter risk management and position sizing",
                metrics_affected=["max_drawdown"]
            ))
        
        # Risk/reward insights
        if metrics.risk_reward_ratio < 1.5:
            insights.append(PerformanceInsight(
                insight_type="poor_risk_reward",
                description=f"Risk/reward ratio is poor ({metrics.risk_reward_ratio:.2f})",
                severity="warning",
                recommendation="Aim for higher reward relative to risk",
                metrics_affected=["risk_reward_ratio"]
            ))
        
        return insights
    
    async def _store_analysis_results(self, metrics: PerformanceMetrics, insights: List[PerformanceInsight]):
        """Store analysis results in database"""
        try:
            async with self.db_connection.get_session() as session:
                # Store performance metrics
                await session.execute("""
                    INSERT INTO performance_metrics (
                        analysis_timestamp, win_rate, profit_factor, sharpe_ratio,
                        max_drawdown, avg_win, avg_loss, total_trades,
                        profitable_trades, losing_trades, total_pnl,
                        max_consecutive_wins, max_consecutive_losses,
                        risk_reward_ratio, expectancy
                    ) VALUES (
                        NOW(), :win_rate, :profit_factor, :sharpe_ratio,
                        :max_drawdown, :avg_win, :avg_loss, :total_trades,
                        :profitable_trades, :losing_trades, :total_pnl,
                        :max_consecutive_wins, :max_consecutive_losses,
                        :risk_reward_ratio, :expectancy
                    )
                """, {
                    'win_rate': metrics.win_rate,
                    'profit_factor': metrics.profit_factor,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'avg_win': metrics.avg_win,
                    'avg_loss': metrics.avg_loss,
                    'total_trades': metrics.total_trades,
                    'profitable_trades': metrics.profitable_trades,
                    'losing_trades': metrics.losing_trades,
                    'total_pnl': metrics.total_pnl,
                    'max_consecutive_wins': metrics.max_consecutive_wins,
                    'max_consecutive_losses': metrics.max_consecutive_losses,
                    'risk_reward_ratio': metrics.risk_reward_ratio,
                    'expectancy': metrics.expectancy
                })
                
                # Store insights
                for insight in insights:
                    await session.execute("""
                        INSERT INTO performance_insights (
                            insight_type, description, severity, recommendation,
                            metrics_affected, timestamp
                        ) VALUES (
                            :insight_type, :description, :severity, :recommendation,
                            :metrics_affected, :timestamp
                        )
                    """, {
                        'insight_type': insight.insight_type,
                        'description': insight.description,
                        'severity': insight.severity,
                        'recommendation': insight.recommendation,
                        'metrics_affected': ','.join(insight.metrics_affected),
                        'timestamp': insight.timestamp
                    })
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"❌ Failed to store analysis results: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance analyzer metrics"""
        return {
            'is_initialized': self.is_initialized,
            'is_running': self.is_running,
            'analysis_interval': self.analysis_interval,
            'min_trades_for_analysis': self.min_trades_for_analysis,
            'performance_history_count': len(self.performance_history),
            'insights_history_count': len(self.insights_history),
            'last_analysis_time': self.performance_history[-1].timestamp if self.performance_history else None
        }
    
    def get_latest_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the latest performance metrics"""
        return self.performance_history[-1] if self.performance_history else None
    
    def get_latest_insights(self, limit: int = 10) -> List[PerformanceInsight]:
        """Get the latest performance insights"""
        return self.insights_history[-limit:] if self.insights_history else []
