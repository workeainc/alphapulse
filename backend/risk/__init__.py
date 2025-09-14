"""
Risk Management Module for AlphaPulse
Week 9: Advanced Risk Management

Provides:
- ML-based position sizing
- Stress testing capabilities
- Compliance reporting
- Audit trail management

Author: AlphaPulse Team
Date: 2025
"""

from .compliance_reporter import ComplianceReporter, ComplianceReport, AuditTrail

__version__ = "1.0.0"
__author__ = "AlphaPulse Team"

__all__ = [
    'ComplianceReporter',
    'ComplianceReport', 
    'AuditTrail'
]
