"""
Outcome Tracking Package for AlphaPulse

This package provides comprehensive outcome tracking for trading signals,
including take profit/stop loss detection, performance analysis, and
automated feedback loops for ML model improvement.
"""

from .outcome_tracker import OutcomeTracker
from .tp_sl_detector import TPSLDetector
from .src.performance_analyzer import PerformanceAnalyzer

__all__ = [
    'OutcomeTracker',
    'TPSLDetector',
    'PerformanceAnalyzer'
]

__version__ = '1.0.0'
