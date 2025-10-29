"""
Signal Outcome Tracking Module

This module tracks hypothetical outcomes of signal recommendations
for ML validation and performance analysis.
"""

from .signal_outcome_tracker import (
    SignalOutcomeTracker,
    SignalOutcome,
    HypotheticalPosition,
    OutcomeTracker
)

# Backward compatibility aliases (DEPRECATED)
import warnings

PaperTradingEngine = SignalOutcomeTracker
PaperTrade = SignalOutcome
PaperPosition = HypotheticalPosition
PaperAccount = OutcomeTracker

warnings.warn(
    "PaperTradingEngine and related classes have been renamed. Use SignalOutcomeTracker instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'SignalOutcomeTracker',
    'SignalOutcome',
    'HypotheticalPosition',
    'OutcomeTracker',
    # Deprecated aliases
    'PaperTradingEngine',
    'PaperTrade',
    'PaperPosition',
    'PaperAccount',
]

