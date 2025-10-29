"""
Prefect Configuration for AlphaPulse Model Retraining
Phase 5: Model Retraining & Continuous Learning
"""

from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any
import os

class PrefectSettings(BaseSettings):
    """Prefect workflow orchestration settings"""
    
    # Prefect Server Configuration
    PREFECT_API_URL: str = "http://localhost:4200/api"
    PREFECT_SERVER_HOST: str = "0.0.0.0"
    PREFECT_SERVER_PORT: int = 4200
    
    # Workflow Settings
    WORKFLOW_NAME: str = "alphapulse-model-retraining"
    WORKFLOW_VERSION: str = "1.0.0"
    
    # Scheduling Configuration (Asia/Dhaka timezone)
    TIMEZONE: str = "Asia/Dhaka"
    
    # Weekly Quick Retrain (Sunday 03:00 Asia/Dhaka)
    WEEKLY_RETRAIN_CRON: str = "0 3 * * 0"  # Sunday 03:00
    WEEKLY_DATA_WEEKS: int = 10  # 8-12 weeks (default 10)
    
    # Monthly Full Retrain (1st of month, 03:00 Asia/Dhaka)
    MONTHLY_RETRAIN_CRON: str = "0 3 1 * *"  # 1st of month, 03:00
    MONTHLY_DATA_MONTHS: int = 18  # 12-24 months (default 18)
    
    # Nightly Incremental Updates (Daily 02:00 Asia/Dhaka)
    NIGHTLY_UPDATE_CRON: str = "0 2 * * *"  # Daily 02:00
    NIGHTLY_DATA_DAYS: int = 1  # 1 day of data
    
    # Resource Configuration
    WEEKLY_RETRAIN_CPU_LIMIT: int = 4
    WEEKLY_RETRAIN_MEMORY_LIMIT: str = "16GB"
    WEEKLY_RETRAIN_TIMEOUT: int = 3600  # 1 hour
    
    MONTHLY_RETRAIN_CPU_LIMIT: int = 8
    MONTHLY_RETRAIN_MEMORY_LIMIT: str = "32GB"
    MONTHLY_RETRAIN_TIMEOUT: int = 14400  # 4 hours
    
    NIGHTLY_UPDATE_CPU_LIMIT: int = 2
    NIGHTLY_UPDATE_MEMORY_LIMIT: str = "8GB"
    NIGHTLY_UPDATE_TIMEOUT: int = 900  # 15 minutes
    
    # Retry Configuration
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 300  # 5 minutes
    
    # Monitoring Configuration
    ENABLE_MONITORING: bool = True
    METRICS_INTERVAL: int = 60  # seconds
    ALERT_EMAIL: Optional[str] = None
    
    # Model Storage
    MODEL_REGISTRY_PATH: str = "models/registry"
    MODEL_BACKUP_PATH: str = "models/backups"
    
    # Data Configuration
    DATA_CACHE_TTL: int = 86400  # 24 hours
    FEATURE_STORE_PATH: str = "data/features"
    
    class Config:
        env_file = ".env"
        env_prefix = "PREFECT_"

# Global settings instance
prefect_settings = PrefectSettings()

# Timezone mapping for cron expressions
TIMEZONE_OFFSETS = {
    "Asia/Dhaka": "+06:00",
    "UTC": "+00:00",
    "America/New_York": "-05:00",
    "Europe/London": "+00:00"
}

# Cron expression templates
CRON_TEMPLATES = {
    "weekly_quick": {
        "description": "Weekly quick retrain (Sunday 03:00 Asia/Dhaka)",
        "cron": "0 3 * * 0",
        "timezone": "Asia/Dhaka",
        "data_window": "8-12 weeks"
    },
    "monthly_full": {
        "description": "Monthly full retrain (1st of month, 03:00 Asia/Dhaka)",
        "cron": "0 3 1 * *",
        "timezone": "Asia/Dhaka",
        "data_window": "12-24 months"
    },
    "nightly_incremental": {
        "description": "Nightly incremental updates (Daily 02:00 Asia/Dhaka)",
        "cron": "0 2 * * *",
        "timezone": "Asia/Dhaka",
        "data_window": "1 day"
    }
}

# Resource profiles for different retraining types
RESOURCE_PROFILES = {
    "weekly_quick": {
        "cpu_limit": 4,
        "memory_limit": "16GB",
        "timeout": 3600,
        "description": "Lightweight retraining for recent trends"
    },
    "monthly_full": {
        "cpu_limit": 8,
        "memory_limit": "32GB",
        "timeout": 14400,
        "description": "Comprehensive retraining for stability"
    },
    "nightly_incremental": {
        "cpu_limit": 2,
        "memory_limit": "8GB",
        "timeout": 900,
        "description": "Minimal updates for daily adaptation"
    }
}
