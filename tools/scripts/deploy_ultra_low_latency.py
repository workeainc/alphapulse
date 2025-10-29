#!/usr/bin/env python3
"""
Ultra-Low Latency System Deployment Script for AlphaPlus
Handles complete deployment including database migrations, dependencies, and service startup
"""

import asyncio
import subprocess
import sys
import os
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UltraLowLatencyDeployment:
    """
    Complete deployment manager for ultra-low latency AlphaPlus system
    """
    
    def __init__(self, config_path: str = "config/deployment_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.project_root = Path(__file__).parent.parent
        self.deployment_log = []
        
        logger.info("🚀 Ultra-Low Latency Deployment Manager initialized")
    
    def _load_config(self) -> Dict:
        """Load deployment configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"✅ Loaded deployment config from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"⚠️ Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default deployment configuration"""
        return {
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "alphapulse",
                "username": "alpha_emon",
                "password": "password"
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
                "password": None
            },
            "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT"],
            "timeframes": ["1m", "5m", "15m", "1h"],
            "performance": {
                "max_workers": 4,
                "confidence_threshold": 0.7,
                "processing_batch_size": 100
            },
            "deployment": {
                "install_dependencies": True,
                "run_migrations": True,
                "create_indexes": True,
                "start_services": True,
                "health_check": True
            }
        }
    
    async def deploy(self):
        """Execute complete deployment process"""
        try:
            logger.info("🚀 Starting Ultra-Low Latency System Deployment...")
            
            # Step 1: Pre-deployment checks
            await self._pre_deployment_checks()
            
            # Step 2: Install dependencies
            if self.config["deployment"]["install_dependencies"]:
                await self._install_dependencies()
            
            # Step 3: Database setup
            if self.config["deployment"]["run_migrations"]:
                await self._setup_database()
            
            # Step 4: Create advanced indexes
            if self.config["deployment"]["create_indexes"]:
                await self._create_advanced_indexes()
            
            # Step 5: Start services
            if self.config["deployment"]["start_services"]:
                await self._start_services()
            
            # Step 6: Health checks
            if self.config["deployment"]["health_check"]:
                await self._health_checks()
            
            # Step 7: Performance validation
            await self._performance_validation()
            
            logger.info("✅ Ultra-Low Latency System Deployment Completed Successfully!")
            self._print_deployment_summary()
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            await self._rollback_deployment()
            raise
    
    async def _pre_deployment_checks(self):
        """Perform pre-deployment checks"""
        logger.info("🔍 Performing pre-deployment checks...")
        
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version < (3, 8):
                raise RuntimeError(f"Python 3.8+ required, found {python_version}")
            logger.info(f"✅ Python version: {python_version}")
            
            # Check required directories
            required_dirs = ["backend", "config", "scripts", "logs"]
            for dir_name in required_dirs:
                if not (self.project_root / dir_name).exists():
                    raise RuntimeError(f"Required directory {dir_name} not found")
            logger.info("✅ Required directories found")
            
            # Check database connectivity
            await self._check_database_connectivity()
            
            # Check Redis connectivity
            await self._check_redis_connectivity()
            
            # Check disk space
            await self._check_disk_space()
            
            # Check system resources
            await self._check_system_resources()
            
            logger.info("✅ All pre-deployment checks passed")
            
        except Exception as e:
            logger.error(f"❌ Pre-deployment checks failed: {e}")
            raise
    
    async def _check_database_connectivity(self):
        """Check database connectivity"""
        try:
            import psycopg2
            from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
            
            db_config = self.config["database"]
            conn = psycopg2.connect(
                host=db_config["host"],
                port=db_config["port"],
                database=db_config["database"],
                user=db_config["username"],
                password=db_config["password"]
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            
            # Test query
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            logger.info(f"✅ Database connectivity: {version.split(',')[0]}")
            
        except Exception as e:
            logger.error(f"❌ Database connectivity check failed: {e}")
            raise
    
    async def _check_redis_connectivity(self):
        """Check Redis connectivity"""
        try:
            import redis
            
            redis_config = self.config["redis"]
            r = redis.Redis(
                host=redis_config["host"],
                port=redis_config["port"],
                password=redis_config["password"],
                decode_responses=True
            )
            
            # Test ping
            response = r.ping()
            if response:
                logger.info("✅ Redis connectivity: OK")
            else:
                raise RuntimeError("Redis ping failed")
                
        except Exception as e:
            logger.error(f"❌ Redis connectivity check failed: {e}")
            raise
    
    async def _check_disk_space(self):
        """Check available disk space"""
        try:
            import shutil
            
            total, used, free = shutil.disk_usage(self.project_root)
            free_gb = free / (1024**3)
            
            if free_gb < 10:  # Require at least 10GB free
                raise RuntimeError(f"Insufficient disk space: {free_gb:.2f}GB free")
            
            logger.info(f"✅ Disk space: {free_gb:.2f}GB free")
            
        except Exception as e:
            logger.error(f"❌ Disk space check failed: {e}")
            raise
    
    async def _check_system_resources(self):
        """Check system resources"""
        try:
            import psutil
            
            # Check CPU cores
            cpu_count = psutil.cpu_count()
            if cpu_count < 4:
                logger.warning(f"⚠️ Low CPU cores: {cpu_count} (recommended: 4+)")
            else:
                logger.info(f"✅ CPU cores: {cpu_count}")
            
            # Check memory
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            if memory_gb < 8:
                logger.warning(f"⚠️ Low memory: {memory_gb:.2f}GB (recommended: 8GB+)")
            else:
                logger.info(f"✅ Memory: {memory_gb:.2f}GB")
            
        except Exception as e:
            logger.error(f"❌ System resources check failed: {e}")
            raise
    
    async def _install_dependencies(self):
        """Install required dependencies"""
        logger.info("📦 Installing dependencies...")
        
        try:
            # Install Python dependencies
            requirements_file = self.project_root / "requirements.txt"
            if requirements_file.exists():
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise RuntimeError(f"Pip install failed: {result.stderr}")
                
                logger.info("✅ Python dependencies installed")
            
            # Install additional ultra-low latency dependencies
            ultra_deps = [
                "uvloop",
                "aioredis",
                "websockets",
                "psycopg2-binary",
                "pandas",
                "numpy",
                "talib-binary"
            ]
            
            for dep in ultra_deps:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", dep
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.warning(f"⚠️ Failed to install {dep}: {result.stderr}")
                else:
                    logger.info(f"✅ Installed {dep}")
            
            logger.info("✅ All dependencies installed")
            
        except Exception as e:
            logger.error(f"❌ Dependency installation failed: {e}")
            raise
    
    async def _setup_database(self):
        """Setup database with migrations"""
        logger.info("🗄️ Setting up database...")
        
        try:
            # Run Alembic migrations
            alembic_cmd = [
                sys.executable, "-m", "alembic", "upgrade", "head"
            ]
            
            result = subprocess.run(
                alembic_cmd,
                cwd=self.project_root / "backend",
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Database migration failed: {result.stderr}")
            
            logger.info("✅ Database migrations completed")
            
            # Verify tables exist
            await self._verify_database_tables()
            
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            raise
    
    async def _verify_database_tables(self):
        """Verify that all required tables exist"""
        try:
            import psycopg2
            
            db_config = self.config["database"]
            conn = psycopg2.connect(
                host=db_config["host"],
                port=db_config["port"],
                database=db_config["database"],
                user=db_config["username"],
                password=db_config["password"]
            )
            
            cursor = conn.cursor()
            
            # Check for required tables
            required_tables = [
                "ultra_low_latency_patterns",
                "ultra_low_latency_signals",
                "ultra_low_latency_performance",
                "shared_memory_buffers"
            ]
            
            for table in required_tables:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    );
                """, (table,))
                
                exists = cursor.fetchone()[0]
                if not exists:
                    raise RuntimeError(f"Required table {table} not found")
                else:
                    logger.info(f"✅ Table {table} exists")
            
            cursor.close()
            conn.close()
            
            logger.info("✅ All required database tables verified")
            
        except Exception as e:
            logger.error(f"❌ Database table verification failed: {e}")
            raise
    
    async def _create_advanced_indexes(self):
        """Create advanced database indexes"""
        logger.info("🔍 Creating advanced indexes...")
        
        try:
            # Run indexing script
            indexing_script = self.project_root / "backend" / "database" / "advanced_indexing.py"
            
            if indexing_script.exists():
                # Import and run indexing
                import sys
                sys.path.append(str(self.project_root / "backend"))
                
                from database.advanced_indexing import AdvancedIndexingManager
                from database.connection import TimescaleDBConnection
                
                # Initialize database connection
                db_url = f"postgresql://{self.config['database']['username']}:{self.config['database']['password']}@{self.config['database']['host']}:{self.config['database']['port']}/{self.config['database']['database']}"
                
                db_connection = TimescaleDBConnection(db_url)
                await db_connection.initialize()
                
                # Create indexes
                indexing_manager = AdvancedIndexingManager(db_connection.get_session_factory())
                await indexing_manager.create_all_advanced_indexes()
                
                await db_connection.close()
                
                logger.info("✅ Advanced indexes created")
            else:
                logger.warning("⚠️ Advanced indexing script not found")
            
        except Exception as e:
            logger.error(f"❌ Advanced indexing failed: {e}")
            raise
    
    async def _start_services(self):
        """Start ultra-low latency services"""
        logger.info("🚀 Starting ultra-low latency services...")
        
        try:
            # Start Redis if not running
            await self._ensure_redis_running()
            
            # Start the integration service
            await self._start_integration_service()
            
            logger.info("✅ Ultra-low latency services started")
            
        except Exception as e:
            logger.error(f"❌ Service startup failed: {e}")
            raise
    
    async def _ensure_redis_running(self):
        """Ensure Redis is running"""
        try:
            import redis
            
            redis_config = self.config["redis"]
            r = redis.Redis(
                host=redis_config["host"],
                port=redis_config["port"],
                password=redis_config["password"],
                decode_responses=True
            )
            
            # Test connection
            r.ping()
            logger.info("✅ Redis is running")
            
        except Exception as e:
            logger.warning(f"⚠️ Redis not accessible: {e}")
            logger.info("💡 Please ensure Redis is running: redis-server")
    
    async def _start_integration_service(self):
        """Start the ultra-low latency integration service"""
        try:
            # Import and start service
            import sys
            sys.path.append(str(self.project_root / "backend"))
            
            from services.ultra_low_latency_integration import (
                UltraLowLatencyIntegrationService, 
                IntegrationConfig
            )
            
            # Create configuration
            config = IntegrationConfig(
                symbols=self.config["symbols"],
                timeframes=self.config["timeframes"],
                redis_url=f"redis://{self.config['redis']['host']}:{self.config['redis']['port']}",
                db_url=f"postgresql://{self.config['database']['username']}:{self.config['database']['password']}@{self.config['database']['host']}:{self.config['database']['port']}/{self.config['database']['database']}",
                max_workers=self.config["performance"]["max_workers"],
                confidence_threshold=self.config["performance"]["confidence_threshold"],
                processing_batch_size=self.config["performance"]["processing_batch_size"]
            )
            
            # Create and start service
            self.integration_service = UltraLowLatencyIntegrationService(config)
            await self.integration_service.initialize()
            
            # Start in background
            asyncio.create_task(self.integration_service.start())
            
            logger.info("✅ Integration service started")
            
        except Exception as e:
            logger.error(f"❌ Integration service startup failed: {e}")
            raise
    
    async def _health_checks(self):
        """Perform health checks"""
        logger.info("🏥 Performing health checks...")
        
        try:
            # Wait for services to stabilize
            await asyncio.sleep(5)
            
            # Check integration service
            if hasattr(self, 'integration_service'):
                stats = await self.integration_service.get_performance_stats()
                
                if stats.get('integration_service', {}).get('is_running'):
                    logger.info("✅ Integration service health: OK")
                else:
                    raise RuntimeError("Integration service not running")
            
            # Check database connectivity
            await self._check_database_connectivity()
            
            # Check Redis connectivity
            await self._check_redis_connectivity()
            
            logger.info("✅ All health checks passed")
            
        except Exception as e:
            logger.error(f"❌ Health checks failed: {e}")
            raise
    
    async def _performance_validation(self):
        """Validate system performance"""
        logger.info("⚡ Validating system performance...")
        
        try:
            if hasattr(self, 'integration_service'):
                # Get performance stats
                stats = await self.integration_service.get_performance_stats()
                
                # Check latency
                avg_latency = stats.get('integration_service', {}).get('avg_processing_latency_ms', 0)
                if avg_latency > 50:  # Should be under 50ms
                    logger.warning(f"⚠️ High average latency: {avg_latency:.2f}ms")
                else:
                    logger.info(f"✅ Average latency: {avg_latency:.2f}ms")
                
                # Check throughput
                messages_processed = stats.get('integration_service', {}).get('total_messages_processed', 0)
                if messages_processed > 0:
                    logger.info(f"✅ Messages processed: {messages_processed}")
                
                # Check pattern detection
                patterns_detected = stats.get('integration_service', {}).get('total_patterns_detected', 0)
                if patterns_detected > 0:
                    logger.info(f"✅ Patterns detected: {patterns_detected}")
                
                # Check signal generation
                signals_generated = stats.get('integration_service', {}).get('total_signals_generated', 0)
                if signals_generated > 0:
                    logger.info(f"✅ Signals generated: {signals_generated}")
            
            logger.info("✅ Performance validation completed")
            
        except Exception as e:
            logger.error(f"❌ Performance validation failed: {e}")
            raise
    
    async def _rollback_deployment(self):
        """Rollback deployment in case of failure"""
        logger.info("🔄 Rolling back deployment...")
        
        try:
            # Stop services
            if hasattr(self, 'integration_service'):
                await self.integration_service.stop()
            
            logger.info("✅ Deployment rollback completed")
            
        except Exception as e:
            logger.error(f"❌ Deployment rollback failed: {e}")
    
    def _print_deployment_summary(self):
        """Print deployment summary"""
        logger.info("\n" + "="*60)
        logger.info("🎉 ULTRA-LOW LATENCY DEPLOYMENT SUMMARY")
        logger.info("="*60)
        logger.info(f"📊 Symbols: {', '.join(self.config['symbols'])}")
        logger.info(f"⏱️ Timeframes: {', '.join(self.config['timeframes'])}")
        logger.info(f"🔧 Max Workers: {self.config['performance']['max_workers']}")
        logger.info(f"🎯 Confidence Threshold: {self.config['performance']['confidence_threshold']}")
        logger.info(f"📦 Processing Batch Size: {self.config['performance']['processing_batch_size']}")
        logger.info("="*60)
        logger.info("✅ System is ready for ultra-low latency trading!")
        logger.info("="*60 + "\n")

async def main():
    """Main deployment function"""
    try:
        # Create deployment manager
        deployment = UltraLowLatencyDeployment()
        
        # Run deployment
        await deployment.deploy()
        
    except KeyboardInterrupt:
        logger.info("🛑 Deployment interrupted by user")
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
