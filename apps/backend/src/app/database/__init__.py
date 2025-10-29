"""
AlphaPlus Database Module
Contains all database models and connection components
"""

from .models import SignalRecommendation, Strategy, MarketData
from .connection import get_db

# Backward compatibility alias
Trade = SignalRecommendation

__all__ = [
    'SignalRecommendation',
    'Trade',  # Deprecated alias
    'Trade',
    'Strategy', 
    'MarketData',
    'get_db'
]
