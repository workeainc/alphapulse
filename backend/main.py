"""
Main Entry Point for AlphaPlus Backend
This file serves as the entry point for Docker deployment
"""

import sys
import os
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from app.main_ai_system_simple import app # Changed to AI system

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False, # Disable reload in production
        log_level="info"
    )
