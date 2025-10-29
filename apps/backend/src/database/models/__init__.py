"""
Database Models Module
Data models and schema definitions
"""

# Import all models from the main models.py file
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    from models import *
except ImportError:
    # If models.py doesn't exist, create empty imports
    pass
