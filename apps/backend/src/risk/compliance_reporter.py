"""
Compliance Reporter Module for AlphaPulse
Week 9: Advanced Risk Management

Features:
- Automated audit trail generation
- Regulatory compliance reporting
- Risk exposure documentation
- Position sizing validation
- Stress test result archiving

Author: AlphaPulse Team
Date: 2025
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
import json
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ComplianceReport:
    """Compliance report structure"""
    report_id: str
    report_type: str
    symbol: str
    timestamp: datetime
    period_start: datetime
    period_end: datetime
    risk_metrics: Dict[str, Any]
    position_data: Dict[str, Any]
    stress_test_results: List[Dict[str, Any]]
    compliance_status: str
    recommendations: List[str]
    metadata: Dict[str, Any]

@dataclass
class AuditTrail:
    """Audit trail entry"""
    entry_id: str
    timestamp: datetime
    action: str
    user: str
    details: Dict[str, Any]
    risk_impact: Optional[float]
    compliance_status: str

class ComplianceReporter:
    """Automated compliance reporting system"""
    
    def __init__(self, db_connection, config: Dict[str, Any] = None):
        self.db = db_connection
        self.config = config or {}
        self.logger = logger
        
        # Compliance configuration
        self.reporting_frequency = self.config.get('reporting_frequency', 'daily')  # daily, weekly, monthly
        self.retention_period = self.config.get('retention_period', 365)  # days
        self.risk_thresholds = self.config.get('risk_thresholds', {
            'max_position_size': 0.1,
            'max_portfolio_risk': 0.02,
            'max_correlation': 0.7,
            'max_concentration': 0.3
        })
        
        # Report storage
        self.reports_dir = Path(self.config.get('reports_dir', 'reports/compliance'))
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Audit trail
        self.audit_trail: List[AuditTrail] = []
        
        # Performance tracking
        self.stats = {
            'reports_generated': 0,
            'audit_entries': 0,
            'compliance_violations': 0,
            'last_report': None
        }
    
    async def generate_compliance_report(self, symbol: str, 
                                       start_time: datetime, 
                                       end_time: datetime,
                                       report_type: str = 'standard') -> ComplianceReport:
        """Generate comprehensive compliance report for a symbol"""
        try:
            self.logger.info(f"Generating compliance report for {symbol} from {start_time} to {end_time}")
            
            # Generate unique report ID
            report_id = f"COMP_{symbol.replace('/', '_')}_{start_time.strftime('%Y%m%d_%H%M%S')}"
            
            # Collect data for the period
            risk_metrics = await self._collect_risk_metrics(symbol, start_time, end_time)
            position_data = await self._collect_position_data(symbol, start_time, end_time)
            stress_test_results = await self._collect_stress_test_results(symbol, start_time, end_time)
            
            # Assess compliance status
            compliance_status = self._assess_compliance_status(risk_metrics, position_data)
            
            # Generate recommendations
            recommendations = self._generate_compliance_recommendations(risk_metrics, position_data)
            
            # Create report
            report = ComplianceReport(
                report_id=report_id,
                report_type=report_type,
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                period_start=start_time,
                period_end=end_time,
                risk_metrics=risk_metrics,
                position_data=position_data,
                stress_test_results=stress_test_results,
                compliance_status=compliance_status,
                recommendations=recommendations,
                metadata={
                    'generator': 'AlphaPulse Compliance Reporter',
                    'version': '1.0.0',
                    'config': self.config
                }
            )
            
            # Store report
            await self._store_compliance_report(report)
            
            # Update statistics
            self.stats['reports_generated'] += 1
            self.stats['last_report'] = datetime.now(timezone.utc)
            
            self.logger.info(f"Compliance report generated: {report_id}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating compliance report: {e}")
            raise
    
    async def _collect_risk_metrics(self, symbol: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Collect risk metrics for the reporting period"""
        try:
            metrics = {}
            
            # Performance metrics
            perf_query = """
                SELECT AVG(pnl) as avg_pnl, 
                       MAX(drawdown) as max_drawdown,
                       AVG(win_rate) as avg_win_rate,
                       COUNT(*) as trade_count
                FROM performance_metrics 
                WHERE symbol = $1 AND timestamp BETWEEN $2 AND $3
            """
            perf_data = await self.db.fetch(perf_query, symbol, start_time, end_time)
            if perf_data:
                metrics['performance'] = {
                    'avg_pnl': float(perf_data[0][0] or 0),
                    'max_drawdown': float(perf_data[0][1] or 0),
                    'avg_win_rate': float(perf_data[0][2] or 0),
                    'trade_count': int(perf_data[0][3] or 0)
                }
            
            # Anomaly metrics
            anomaly_query = """
                SELECT COUNT(*) as anomaly_count,
                       AVG(z_score) as avg_z_score,
                       MAX(z_score) as max_z_score
                FROM anomalies 
                WHERE symbol = $1 AND timestamp BETWEEN $2 AND $3
            """
            anomaly_data = await self.db.fetch(anomaly_query, symbol, start_time, end_time)
            if anomaly_data:
                metrics['anomalies'] = {
                    'anomaly_count': int(anomaly_data[0][0] or 0),
                    'avg_z_score': float(anomaly_data[0][1] or 0),
                    'max_z_score': float(anomaly_data[0][2] or 0)
                }
            
            # System metrics
            system_query = """
                SELECT metric_name, AVG(metric_value) as avg_value
                FROM system_metrics 
                WHERE timestamp BETWEEN $1 AND $2
                GROUP BY metric_name
            """
            system_data = await self.db.fetch(system_query, start_time, end_time)
            if system_data:
                metrics['system'] = {
                    row[0]: float(row[1] or 0) for row in system_data
                }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting risk metrics: {e}")
            return {}
    
    async def _collect_position_data(self, symbol: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Collect position data for the reporting period"""
        try:
            position_data = {}
            
            # Signal predictions
            signal_query = """
                SELECT AVG(confidence) as avg_confidence,
                       AVG(predicted_pnl) as avg_predicted_pnl,
                       COUNT(*) as signal_count
                FROM signal_predictions 
                WHERE symbol = $1 AND timestamp BETWEEN $2 AND $3
            """
            signal_data = await self.db.fetch(signal_query, symbol, start_time, end_time)
            if signal_data:
                position_data['signals'] = {
                    'avg_confidence': float(signal_data[0][0] or 0),
                    'avg_predicted_pnl': float(signal_data[0][1] or 0),
                    'signal_count': int(signal_data[0][2] or 0)
                }
            
            # Funding rates
            funding_query = """
                SELECT AVG(funding_rate) as avg_funding_rate,
                       MAX(funding_rate) as max_funding_rate,
                       MIN(funding_rate) as min_funding_rate
                FROM funding_rates 
                WHERE symbol = $1 AND timestamp BETWEEN $2 AND $3
            """
            funding_data = await self.db.fetch(funding_query, symbol, start_time, end_time)
            if funding_data:
                position_data['funding_rates'] = {
                    'avg_funding_rate': float(funding_data[0][0] or 0),
                    'max_funding_rate': float(funding_data[0][1] or 0),
                    'min_funding_rate': float(funding_data[0][2] or 0)
                }
            
            return position_data
            
        except Exception as e:
            self.logger.error(f"Error collecting position data: {e}")
            return {}
    
    async def _collect_stress_test_results(self, symbol: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Collect stress test results for the reporting period"""
        try:
            # For now, return empty list as stress test results are stored in memory
            # In a production system, these would be stored in the database
            return []
            
        except Exception as e:
            self.logger.error(f"Error collecting stress test results: {e}")
            return []
    
    def _assess_compliance_status(self, risk_metrics: Dict[str, Any], position_data: Dict[str, Any]) -> str:
        """Assess overall compliance status"""
        try:
            violations = []
            
            # Check performance compliance
            if 'performance' in risk_metrics:
                perf = risk_metrics['performance']
                if perf.get('max_drawdown', 0) > self.risk_thresholds['max_portfolio_risk']:
                    violations.append('max_drawdown_exceeded')
                
                if perf.get('trade_count', 0) < 10:  # Minimum trade count for reliable metrics
                    violations.append('insufficient_trade_data')
            
            # Check anomaly compliance
            if 'anomalies' in risk_metrics:
                anomalies = risk_metrics['anomalies']
                if anomalies.get('max_z_score', 0) > 5.0:  # Excessive anomalies
                    violations.append('excessive_anomalies')
            
            # Check signal compliance
            if 'signals' in position_data:
                signals = position_data['signals']
                if signals.get('avg_confidence', 0) < 0.6:  # Low confidence signals
                    violations.append('low_signal_confidence')
            
            # Determine compliance status
            if not violations:
                return 'compliant'
            elif len(violations) <= 2:
                return 'minor_violations'
            else:
                return 'major_violations'
                
        except Exception as e:
            self.logger.error(f"Error assessing compliance status: {e}")
            return 'unknown'
    
    def _generate_compliance_recommendations(self, risk_metrics: Dict[str, Any], position_data: Dict[str, Any]) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        # Performance-based recommendations
        if 'performance' in risk_metrics:
            perf = risk_metrics['performance']
            if perf.get('max_drawdown', 0) > self.risk_thresholds['max_portfolio_risk']:
                recommendations.append("Reduce position sizes to limit maximum drawdown")
            
            if perf.get('avg_win_rate', 0) < 0.5:
                recommendations.append("Review signal generation strategy to improve win rate")
        
        # Anomaly-based recommendations
        if 'anomalies' in risk_metrics:
            anomalies = risk_metrics['anomalies']
            if anomalies.get('anomaly_count', 0) > 10:
                recommendations.append("Investigate increased anomaly frequency")
        
        # Signal-based recommendations
        if 'signals' in position_data:
            signals = position_data['signals']
            if signals.get('avg_confidence', 0) < 0.6:
                recommendations.append("Improve signal quality through enhanced feature engineering")
        
        # Default recommendation if no specific issues
        if not recommendations:
            recommendations.append("Current operations appear compliant with risk parameters")
        
        return recommendations
    
    async def _store_compliance_report(self, report: ComplianceReport):
        """Store compliance report for future reference"""
        try:
            # Convert to dictionary for storage
            report_dict = asdict(report)
            
            # Store in database if table exists
            try:
                await self.db.execute("""
                    INSERT INTO compliance_reports (
                        report_id, report_type, symbol, timestamp, period_start, period_end,
                        risk_metrics, position_data, stress_test_results, compliance_status,
                        recommendations, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """, report.report_id, report.report_type, report.symbol, report.timestamp,
                     report.period_start, report.period_end, json.dumps(report.risk_metrics),
                     json.dumps(report.position_data), json.dumps(report.stress_test_results),
                     report.compliance_status, json.dumps(report.recommendations),
                     json.dumps(report.metadata))
                
                self.logger.info(f"Compliance report stored in database: {report.report_id}")
                
            except Exception as db_error:
                self.logger.warning(f"Could not store report in database: {db_error}")
            
            # Store as JSON file
            report_file = self.reports_dir / f"{report.report_id}.json"
            with open(report_file, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            self.logger.info(f"Compliance report saved to file: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Error storing compliance report: {e}")
    
    async def add_audit_entry(self, action: str, user: str, details: Dict[str, Any], 
                             risk_impact: Optional[float] = None) -> str:
        """Add entry to audit trail"""
        try:
            entry_id = f"AUDIT_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Determine compliance status based on risk impact
            if risk_impact is None:
                compliance_status = 'unknown'
            elif risk_impact < 0.01:
                compliance_status = 'low_risk'
            elif risk_impact < 0.05:
                compliance_status = 'medium_risk'
            else:
                compliance_status = 'high_risk'
            
            audit_entry = AuditTrail(
                entry_id=entry_id,
                timestamp=datetime.now(timezone.utc),
                action=action,
                user=user,
                details=details,
                risk_impact=risk_impact,
                compliance_status=compliance_status
            )
            
            # Add to audit trail
            self.audit_trail.append(audit_entry)
            
            # Store in database if table exists
            try:
                await self.db.execute("""
                    INSERT INTO audit_trail (
                        entry_id, timestamp, action, user, details, risk_impact, compliance_status
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, entry_id, audit_entry.timestamp, action, user, 
                     json.dumps(details), risk_impact, compliance_status)
                
            except Exception as db_error:
                self.logger.warning(f"Could not store audit entry in database: {db_error}")
            
            # Update statistics
            self.stats['audit_entries'] += 1
            if compliance_status in ['medium_risk', 'high_risk']:
                self.stats['compliance_violations'] += 1
            
            self.logger.info(f"Audit entry added: {entry_id} - {action}")
            return entry_id
            
        except Exception as e:
            self.logger.error(f"Error adding audit entry: {e}")
            raise
    
    async def get_compliance_summary(self, symbol: str = None, days: int = 30) -> Dict[str, Any]:
        """Get compliance summary for the specified period"""
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=days)
            
            summary = {
                'period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'days': days
                },
                'statistics': self.stats,
                'compliance_status': 'unknown'
            }
            
            # Get recent reports
            if symbol:
                # In a production system, this would query the database
                summary['symbol'] = symbol
                summary['recent_reports'] = []
            else:
                summary['recent_reports'] = []
            
            # Get audit trail summary
            recent_audit = [entry for entry in self.audit_trail 
                           if entry.timestamp >= start_time]
            
            summary['audit_summary'] = {
                'total_entries': len(recent_audit),
                'risk_distribution': {
                    'low_risk': len([e for e in recent_audit if e.compliance_status == 'low_risk']),
                    'medium_risk': len([e for e in recent_audit if e.compliance_status == 'medium_risk']),
                    'high_risk': len([e for e in recent_audit if e.compliance_status == 'high_risk'])
                },
                'recent_actions': [
                    {
                        'action': entry.action,
                        'user': entry.user,
                        'timestamp': entry.timestamp.isoformat(),
                        'compliance_status': entry.compliance_status
                    }
                    for entry in recent_audit[-10:]  # Last 10 entries
                ]
            }
            
            # Determine overall compliance status
            high_risk_count = summary['audit_summary']['risk_distribution']['high_risk']
            if high_risk_count == 0:
                summary['compliance_status'] = 'compliant'
            elif high_risk_count <= 2:
                summary['compliance_status'] = 'minor_violations'
            else:
                summary['compliance_status'] = 'major_violations'
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting compliance summary: {e}")
            return {'error': str(e)}
    
    async def close(self):
        """Close the compliance reporter"""
        try:
            self.logger.info("Compliance Reporter closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing Compliance Reporter: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
