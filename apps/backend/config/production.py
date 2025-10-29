"""Production Configuration for AlphaPulse - Enterprise Edition"""

import os
import logging
from typing import Dict, Any, Optional, List

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ProductionConfig:
    """Production-specific configuration settings"""
    
    # Environment
    ENVIRONMENT = "production"
    DEBUG = False
    LOG_LEVEL = "INFO"
    
    # Database Configuration
    DATABASE_CONFIG = {
        "host": os.getenv("TIMESCALEDB_HOST", "localhost"),
        "port": int(os.getenv("TIMESCALEDB_PORT", 5432)),
        "database": os.getenv("TIMESCALEDB_DATABASE", "alphapulse"),
        "username": os.getenv("TIMESCALEDB_USERNAME", "alpha_emon"),
        "password": os.getenv("TIMESCALEDB_PASSWORD", "Emon_@17711"),
        "pool_size": int(os.getenv("DB_POOL_SIZE", 50)),
        "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", 100)),
        "pool_timeout": int(os.getenv("DB_POOL_TIMEOUT", 60)),
        "pool_recycle": int(os.getenv("DB_POOL_RECYCLE", 3600)),
        "echo": os.getenv("DB_ECHO", "false").lower() == "true",
        "ssl_mode": os.getenv("DB_SSL_MODE", "prefer"),
        "application_name": os.getenv("DB_APP_NAME", "AlphaPulse-Prod")
    }
    
    # Redis Configuration
    REDIS_CONFIG = {
        "host": os.getenv("REDIS_HOST", "localhost"),
        "port": int(os.getenv("REDIS_PORT", 6379)),
        "db": int(os.getenv("REDIS_DB", 0)),
        "password": os.getenv("REDIS_PASSWORD"),
        "max_connections": int(os.getenv("REDIS_MAX_CONNECTIONS", 100)),
        "socket_timeout": int(os.getenv("REDIS_SOCKET_TIMEOUT", 5)),
        "socket_connect_timeout": int(os.getenv("REDIS_CONNECT_TIMEOUT", 5)),
        "retry_on_timeout": os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true",
        "health_check_interval": int(os.getenv("REDIS_HEALTH_CHECK", 30)),
        "decode_responses": True
    }
    
    # Server Configuration
    SERVER_CONFIG = {
        "host": os.getenv("API_HOST", "0.0.0.0"),
        "port": int(os.getenv("API_PORT", 8000)),
        "workers": int(os.getenv("WORKERS", 4)),
        "worker_class": os.getenv("WORKER_CLASS", "uvicorn.workers.UvicornWorker"),
        "max_requests": int(os.getenv("MAX_REQUESTS", 1000)),
        "max_requests_jitter": int(os.getenv("MAX_REQUESTS_JITTER", 100)),
        "timeout": int(os.getenv("TIMEOUT", 120)),
        "keepalive": int(os.getenv("KEEPALIVE", 2)),
        "backlog": int(os.getenv("BACKLOG", 2048))
    }
    
    # Real-time Configuration
    REAL_TIME_CONFIG = {
        "websocket_enabled": os.getenv("WEBSOCKET_ENABLED", "true").lower() == "true",
        "websocket_port": int(os.getenv("WEBSOCKET_PORT", 8001)),
        "max_connections": int(os.getenv("MAX_WEBSOCKET_CONNECTIONS", 1000)),
        "heartbeat_interval": int(os.getenv("HEARTBEAT_INTERVAL", 30)),
        "connection_timeout": int(os.getenv("CONNECTION_TIMEOUT", 300)),
        "message_queue_size": int(os.getenv("MESSAGE_QUEUE_SIZE", 10000)),
        "broadcast_interval": int(os.getenv("BROADCAST_INTERVAL", 1)),
        "data_streams": {
            "market_data": True,
            "signals": True,
            "alerts": True,
            "performance": True,
            "system_metrics": True
        }
    }
    
    # Monitoring Configuration
    MONITORING_CONFIG = {
        "enabled": os.getenv("MONITORING_ENABLED", "true").lower() == "true",
        "metrics_port": int(os.getenv("METRICS_PORT", 9090)),
        "health_check_port": int(os.getenv("HEALTH_CHECK_PORT", 8080)),
        "prometheus_enabled": os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true",
        "grafana_enabled": os.getenv("GRAFANA_ENABLED", "true").lower() == "true",
        "alerting_enabled": os.getenv("ALERTING_ENABLED", "true").lower() == "true",
        "metrics_interval": int(os.getenv("METRICS_INTERVAL", 60)),
        "health_check_interval": int(os.getenv("HEALTH_CHECK_INTERVAL", 30)),
        "log_metrics": os.getenv("LOG_METRICS", "true").lower() == "true"
    }
    
    # Alerting Configuration
    ALERTING_CONFIG = {
        "slack_webhook": os.getenv("SLACK_WEBHOOK_URL"),
        "email_smtp_host": os.getenv("EMAIL_SMTP_HOST"),
        "email_smtp_port": int(os.getenv("EMAIL_SMTP_PORT", 587)),
        "email_username": os.getenv("EMAIL_USERNAME"),
        "email_password": os.getenv("EMAIL_PASSWORD"),
        "email_recipients": os.getenv("EMAIL_RECIPIENTS", "").split(",") if os.getenv("EMAIL_RECIPIENTS") else [],
        "alert_levels": ["critical", "warning", "info"],
        "alert_cooldown": int(os.getenv("ALERT_COOLDOWN", 300))
    }
    
    # Trading Configuration
    TRADING_CONFIG = {
        "enabled": os.getenv("TRADING_ENABLED", "false").lower() == "true",
        "paper_trading": os.getenv("PAPER_TRADING", "true").lower() == "true",
        "max_symbols": int(os.getenv("MAX_SYMBOLS", 3000)),
        "max_portfolio_risk": float(os.getenv("MAX_PORTFOLIO_RISK", 0.1)),
        "max_position_size": float(os.getenv("MAX_POSITION_SIZE", 0.05)),
        "stop_loss_percentage": float(os.getenv("STOP_LOSS_PERCENTAGE", 0.02)),
        "take_profit_percentage": float(os.getenv("TAKE_PROFIT_PERCENTAGE", 0.04)),
        "max_open_positions": int(os.getenv("MAX_OPEN_POSITIONS", 10)),
        "position_sizing_method": os.getenv("POSITION_SIZING_METHOD", "kelly")
    }
    
    # Performance Configuration
    PERFORMANCE_CONFIG = {
        "cache_enabled": os.getenv("CACHE_ENABLED", "true").lower() == "true",
        "cache_ttl": int(os.getenv("CACHE_TTL", 300)),
        "cache_max_size": int(os.getenv("CACHE_MAX_SIZE", 10000)),
        "async_processing": os.getenv("ASYNC_PROCESSING", "true").lower() == "true",
        "max_concurrent_tasks": int(os.getenv("MAX_CONCURRENT_TASKS", 100)),
        "task_timeout": int(os.getenv("TASK_TIMEOUT", 300)),
        "batch_size": int(os.getenv("BATCH_SIZE", 1000)),
        "compression_enabled": os.getenv("COMPRESSION_ENABLED", "true").lower() == "true"
    }
    
    # Security Configuration
    SECURITY_CONFIG = {
        "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
        "cors_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "cors_headers": ["*"],
        "rate_limit_enabled": os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
        "rate_limit_requests": int(os.getenv("RATE_LIMIT_REQUESTS", 100)),
        "rate_limit_window": int(os.getenv("RATE_LIMIT_WINDOW", 60)),
        "jwt_secret": os.getenv("JWT_SECRET", "your-secret-key"),
        "jwt_algorithm": os.getenv("JWT_ALGORITHM", "HS256"),
        "jwt_expiration": int(os.getenv("JWT_EXPIRATION", 3600)),
        "ssl_enabled": os.getenv("SSL_ENABLED", "false").lower() == "true",
        "ssl_cert_path": os.getenv("SSL_CERT_PATH"),
        "ssl_key_path": os.getenv("SSL_KEY_PATH")
    }
    
    # Logging Configuration
    LOGGING_CONFIG = {
        "level": os.getenv("LOG_LEVEL", "INFO"),
        "format": os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        "file_path": os.getenv("LOG_FILE_PATH", "logs/production.log"),
        "max_size": int(os.getenv("LOG_MAX_SIZE", 10485760)),  # 10MB
        "backup_count": int(os.getenv("LOG_BACKUP_COUNT", 5)),
        "json_format": os.getenv("LOG_JSON_FORMAT", "false").lower() == "true",
        "include_timestamp": os.getenv("LOG_INCLUDE_TIMESTAMP", "true").lower() == "true"
    }
    
    # External Services Configuration
    EXTERNAL_SERVICES_CONFIG = {
        "alpha_vantage_api_key": os.getenv("ALPHA_VANTAGE_API_KEY"),
        "polygon_api_key": os.getenv("POLYGON_API_KEY"),
        "news_api_key": os.getenv("NEWS_API_KEY"),
        "alpha_vantage_base_url": os.getenv("ALPHA_VANTAGE_BASE_URL", "https://www.alphavantage.co/query"),
        "polygon_base_url": os.getenv("POLYGON_BASE_URL", "https://api.polygon.io"),
        "news_api_base_url": os.getenv("NEWS_API_BASE_URL", "https://newsapi.org/v2"),
        "alpha_vantage_rate_limit": int(os.getenv("ALPHA_VANTAGE_RATE_LIMIT", 5)),
        "polygon_rate_limit": int(os.getenv("POLYGON_RATE_LIMIT", 5)),
        "news_api_rate_limit": int(os.getenv("NEWS_API_RATE_LIMIT", 1)),
        "api_timeout": int(os.getenv("API_TIMEOUT", 30)),
        "retry_attempts": int(os.getenv("RETRY_ATTEMPTS", 3)),
        "retry_delay": int(os.getenv("RETRY_DELAY", 1))
    }
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get database URL for production"""
        config = cls.DATABASE_CONFIG
        return f"postgresql://{config['username']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
    
    @classmethod
    def get_redis_url(cls) -> str:
        """Get Redis URL for production"""
        config = cls.REDIS_CONFIG
        if config['password']:
            return f"redis://:{config['password']}@{config['host']}:{config['port']}/{config['db']}"
        return f"redis://{config['host']}:{config['port']}/{config['db']}"
    
    @classmethod
    def get_websocket_url(cls) -> str:
        """Get WebSocket URL for production"""
        config = cls.REAL_TIME_CONFIG
        protocol = "wss" if cls.SECURITY_CONFIG['ssl_enabled'] else "ws"
        return f"{protocol}://{cls.SERVER_CONFIG['host']}:{config['websocket_port']}"
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate production configuration"""
        issues = {
            "database": {"valid": True, "issues": []},
            "redis": {"valid": True, "issues": []},
            "security": {"valid": True, "issues": []},
            "monitoring": {"valid": True, "issues": []},
            "trading": {"valid": True, "issues": []},
            "external_services": {"valid": True, "issues": []}
        }
        
        # Validate database configuration
        if not cls.DATABASE_CONFIG['host']:
            issues["database"]["valid"] = False
            issues["database"]["issues"].append("Database host not configured")
        
        # Validate Redis configuration
        if not cls.REDIS_CONFIG['host']:
            issues["redis"]["valid"] = False
            issues["redis"]["issues"].append("Redis host not configured")
        
        # Validate security configuration
        if cls.SECURITY_CONFIG['jwt_secret'] == "your-secret-key":
            issues["security"]["valid"] = False
            issues["security"]["issues"].append("JWT secret not configured")
        
        # Validate external services
        if not cls.EXTERNAL_SERVICES_CONFIG['alpha_vantage_api_key']:
            issues["external_services"]["issues"].append("Alpha Vantage API key not configured")
        
        return issues
    
    @classmethod
    def get_health_check_config(cls) -> Dict[str, Any]:
        """Get health check configuration"""
        return {
            "database": {
                "enabled": True,
                "interval": 30,
                "timeout": 10
            },
            "redis": {
                "enabled": True,
                "interval": 30,
                "timeout": 5
            },
            "external_apis": {
                "enabled": True,
                "interval": 60,
                "timeout": 15
            },
            "system_metrics": {
                "enabled": True,
                "interval": 60,
                "timeout": 5
            }
        }
    
    @classmethod
    def get_metrics_config(cls) -> Dict[str, Any]:
        """Get metrics configuration"""
        return {
            "enabled": cls.MONITORING_CONFIG['enabled'],
            "port": cls.MONITORING_CONFIG['metrics_port'],
            "interval": cls.MONITORING_CONFIG['metrics_interval'],
            "prometheus": {
                "enabled": cls.MONITORING_CONFIG['prometheus_enabled'],
                "path": "/metrics"
            },
            "custom_metrics": {
                "signal_accuracy": True,
                "processing_latency": True,
                "cache_hit_rate": True,
                "error_rate": True,
                "throughput": True
            }
        }

# Create production config instance
production_config = ProductionConfig()

# Log configuration initialization
logger.info("Production configuration initialized successfully")
logger.info(f"Environment: {production_config.ENVIRONMENT}")
logger.info(f"Database: {production_config.DATABASE_CONFIG['host']}:{production_config.DATABASE_CONFIG['port']}")
logger.info(f"Redis: {production_config.REDIS_CONFIG['host']}:{production_config.REDIS_CONFIG['port']}")
logger.info(f"Server: {production_config.SERVER_CONFIG['host']}:{production_config.SERVER_CONFIG['port']}")
logger.info(f"WebSocket: {production_config.get_websocket_url()}")
logger.info(f"Monitoring: {production_config.MONITORING_CONFIG['enabled']}")
logger.info(f"Trading: {production_config.TRADING_CONFIG['enabled']}")
