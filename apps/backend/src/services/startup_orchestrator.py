"""
Startup Orchestrator for AlphaPulse
Coordinates initialization and startup of all services for 100-symbol system
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import asyncpg
import yaml
import os

from src.services.dynamic_symbol_manager import DynamicSymbolManager, symbol_manager
from src.services.websocket_orchestrator import WebSocketOrchestrator
from src.services.signal_generation_scheduler import SignalGenerationScheduler
from src.data.realtime_data_pipeline import RealTimeDataPipeline
from src.services.ai_model_integration_service import AIModelIntegrationService
from src.services.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class StartupOrchestrator:
    """
    Orchestrates the startup sequence for scaled AlphaPulse system
    Ensures proper initialization order and dependency management
    """
    
    def __init__(self, config_path: str = "config/symbol_config.yaml"):
        self.config = self._load_config(config_path)
        self.logger = logger
        
        # Services (will be initialized)
        self.db_pool: Optional[asyncpg.Pool] = None
        self.symbol_manager: Optional[DynamicSymbolManager] = None
        self.data_pipeline: Optional[RealTimeDataPipeline] = None
        self.websocket_orchestrator: Optional[WebSocketOrchestrator] = None
        self.signal_scheduler: Optional[SignalGenerationScheduler] = None
        self.ai_service: Optional[AIModelIntegrationService] = None
        
        # State
        self.is_initialized = False
        self.startup_time: Optional[datetime] = None
        self.startup_duration_seconds: float = 0.0
        
        logger.info("✅ Startup Orchestrator created")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file (legacy single-file loading)"""
        try:
            full_path = os.path.join(os.path.dirname(__file__), '..', '..', config_path)
            if os.path.exists(full_path):
                with open(full_path, 'r') as f:
                    config = yaml.safe_load(f)
                    logger.info(f"✅ Loaded configuration from {config_path}")
                    
                    # Also load MTF config using ConfigLoader
                    config_loader = ConfigLoader()
                    full_config = config_loader.load_all_configs()
                    
                    # Merge with loaded config
                    merged = {**full_config, **config}
                    logger.info("✅ Merged all configuration files")
                    return merged
        except Exception as e:
            logger.warning(f"⚠️ Could not load config: {e}, using defaults")
        
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'symbol_management': {
                'total_symbols': 100,
                'futures_count': 50,
                'spot_count': 50,
                'update_interval_hours': 24
            },
            'websocket': {
                'max_connections': 2,
                'streams_per_connection': 100
            },
            'signal_generation': {
                'symbols_per_batch': 10,
                'analysis_interval_seconds': 60
            },
            'database': {
                'min_pool_size': 10,
                'max_pool_size': 30
            }
        }
    
    async def startup(self, database_url: str):
        """
        Execute complete startup sequence
        Returns True if successful, False otherwise
        """
        startup_start = datetime.now(timezone.utc)
        self.startup_time = startup_start
        
        try:
            logger.info("=" * 80)
            logger.info("🚀 ALPHAPULSE STARTUP SEQUENCE - 100 SYMBOL ORCHESTRATION")
            logger.info("=" * 80)
            
            # Phase 1: Database Connection
            logger.info("\n[1/6] 🗄️ Initializing database connection pool...")
            await self._initialize_database(database_url)
            logger.info("✅ Database connection pool ready")
            
            # Phase 2: Symbol List Manager
            logger.info("\n[2/6] 📋 Initializing dynamic symbol manager...")
            await self._initialize_symbol_manager()
            logger.info("✅ Symbol manager ready with symbol list")
            
            # Phase 3: Data Pipeline
            logger.info("\n[3/6] 🔄 Initializing real-time data pipeline...")
            await self._initialize_data_pipeline()
            logger.info("✅ Data pipeline ready")
            
            # Phase 4: WebSocket Orchestrator
            logger.info("\n[4/6] 📡 Initializing WebSocket orchestrator...")
            await self._initialize_websocket_orchestrator()
            logger.info("✅ WebSocket orchestrator ready with connections")
            
            # Phase 5: AI Model Service
            logger.info("\n[5/6] 🧠 Initializing AI model integration service...")
            await self._initialize_ai_service()
            logger.info("✅ AI service ready")
            
            # Phase 6: Signal Generation Scheduler
            logger.info("\n[6/6] ⏰ Initializing signal generation scheduler...")
            await self._initialize_signal_scheduler()
            logger.info("✅ Signal scheduler ready")
            
            # Mark as initialized
            self.is_initialized = True
            self.startup_duration_seconds = (datetime.now(timezone.utc) - startup_start).total_seconds()
            
            logger.info("\n" + "=" * 80)
            logger.info(f"✅ STARTUP COMPLETE in {self.startup_duration_seconds:.2f}s")
            logger.info("=" * 80)
            logger.info(f"📊 System Status:")
            logger.info(f"   - Symbols tracked: {len(await self.symbol_manager.get_active_symbols())}")
            logger.info(f"   - WebSocket connections: {len(self.websocket_orchestrator.clients)}")
            logger.info(f"   - Database pool: {self.config['database']['min_pool_size']}-{self.config['database']['max_pool_size']} connections")
            logger.info(f"   - Analysis interval: Every {self.config['signal_generation']['analysis_interval_seconds']}s")
            logger.info("=" * 80 + "\n")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ STARTUP FAILED: {e}")
            await self.shutdown()
            return False
    
    async def _initialize_database(self, database_url: str):
        """Initialize database connection pool"""
        try:
            self.db_pool = await asyncpg.create_pool(
                database_url,
                min_size=self.config['database']['min_pool_size'],
                max_size=self.config['database']['max_pool_size'],
                command_timeout=60
            )
            
            # Test connection
            async with self.db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    async def _initialize_symbol_manager(self):
        """Initialize symbol manager and load/update symbol list"""
        try:
            self.symbol_manager = symbol_manager
            await self.symbol_manager.initialize(self.db_pool)
            
            # Check if we need to update symbol list
            if await self.symbol_manager.should_update():
                logger.info("🔄 Updating symbol list from Binance...")
                await self.symbol_manager.update_symbol_list()
            
            # Start auto-update background task
            asyncio.create_task(self.symbol_manager.auto_update_loop())
            
        except Exception as e:
            logger.error(f"❌ Symbol manager initialization failed: {e}")
            raise
    
    async def _initialize_data_pipeline(self):
        """Initialize real-time data pipeline"""
        try:
            # Use Docker port 55433 for TimescaleDB, 56379 for Redis
            db_url = os.getenv('DATABASE_URL', 'postgresql://alpha_emon:Emon_%4017711@localhost:55433/alphapulse')
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', '56379'))
            
            self.data_pipeline = RealTimeDataPipeline(
                redis_host=redis_host,
                redis_port=redis_port,
                db_url=db_url
            )
            await self.data_pipeline.initialize()
            
        except Exception as e:
            logger.error(f"❌ Data pipeline initialization failed: {e}")
            raise
    
    async def _initialize_websocket_orchestrator(self):
        """Initialize WebSocket orchestrator with connections"""
        try:
            self.websocket_orchestrator = WebSocketOrchestrator(
                symbol_manager=self.symbol_manager,
                data_pipeline=self.data_pipeline,
                config=self.config
            )
            
            await self.websocket_orchestrator.initialize()
            await self.websocket_orchestrator.start()
            
        except Exception as e:
            logger.error(f"❌ WebSocket orchestrator initialization failed: {e}")
            raise
    
    async def _initialize_ai_service(self):
        """Initialize AI model integration service"""
        try:
            self.ai_service = AIModelIntegrationService()
            
        except Exception as e:
            logger.error(f"❌ AI service initialization failed: {e}")
            raise
    
    async def _initialize_signal_scheduler(self):
        """Initialize signal generation scheduler"""
        try:
            self.signal_scheduler = SignalGenerationScheduler(
                symbol_manager=self.symbol_manager,
                ai_service=self.ai_service,
                config=self.config
            )
            
            await self.signal_scheduler.initialize()
            await self.signal_scheduler.start()
            
        except Exception as e:
            logger.error(f"❌ Signal scheduler initialization failed: {e}")
            raise
    
    async def shutdown(self):
        """Gracefully shutdown all services"""
        logger.info("🛑 Initiating graceful shutdown...")
        
        try:
            # Stop signal scheduler
            if self.signal_scheduler:
                await self.signal_scheduler.stop()
            
            # Stop WebSocket orchestrator
            if self.websocket_orchestrator:
                await self.websocket_orchestrator.stop()
            
            # Close data pipeline
            if self.data_pipeline:
                await self.data_pipeline.close()
            
            # Close database pool
            if self.db_pool:
                await self.db_pool.close()
            
            logger.info("✅ Shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        if not self.is_initialized:
            return {
                'initialized': False,
                'error': 'System not initialized'
            }
        
        try:
            # Get status from all components
            symbol_stats = self.symbol_manager.get_stats()
            ws_stats = self.websocket_orchestrator.get_stats()
            scheduler_stats = self.signal_scheduler.get_stats()
            ws_health = await self.websocket_orchestrator.get_health_status()
            
            return {
                'initialized': True,
                'startup_time': self.startup_time.isoformat(),
                'uptime_seconds': (datetime.now(timezone.utc) - self.startup_time).total_seconds(),
                'startup_duration_seconds': self.startup_duration_seconds,
                'symbol_manager': symbol_stats,
                'websocket': {
                    'stats': ws_stats.__dict__,
                    'health': ws_health
                },
                'signal_scheduler': scheduler_stats,
                'database': {
                    'pool_size': self.db_pool.get_size() if self.db_pool else 0,
                    'pool_max': self.config['database']['max_pool_size']
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting system status: {e}")
            return {
                'initialized': self.is_initialized,
                'error': str(e)
            }

# Global instance
startup_orchestrator = StartupOrchestrator()

