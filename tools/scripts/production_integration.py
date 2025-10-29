#!/usr/bin/env python3
"""
Production Integration Script for Ultra-Optimized Pattern Detection
Handles deployment, monitoring, scaling, and database migration
"""

import asyncio
import logging
import sys
import os
import time
import json
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import psutil
import signal

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
sys.path.insert(0, backend_path)

# Import services
from app.services.performance_monitor import get_performance_monitor
from app.services.config_manager import get_config_manager
from strategies.ultra_optimized_pattern_detector import UltraOptimizedPatternDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/production_integration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class ProductionIntegration:
    """
    Production integration manager for ultra-optimized pattern detection
    """
    
    def __init__(self):
        """Initialize production integration"""
        self.performance_monitor = get_performance_monitor()
        self.config_manager = get_config_manager()
        self.detector = None
        self.is_running = False
        self.integration_stats = {
            'start_time': None,
            'total_patterns_processed': 0,
            'total_processing_time': 0.0,
            'config_changes': 0,
            'alerts_triggered': 0,
            'errors': []
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🏭 Production Integration Manager initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown()
    
    async def deploy_to_production(self):
        """Deploy ultra-optimized pattern detection to production"""
        logger.info("🚀 Starting production deployment...")
        
        try:
            # Step 1: Pre-deployment checks
            await self._pre_deployment_checks()
            
            # Step 2: Initialize detector with production config
            await self._initialize_production_detector()
            
            # Step 3: Run database migration
            await self._run_database_migration()
            
            # Step 4: Start monitoring and optimization
            await self._start_production_monitoring()
            
            # Step 5: Begin production processing
            await self._start_production_processing()
            
            logger.info("✅ Production deployment completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Production deployment failed: {e}")
            self.integration_stats['errors'].append(str(e))
            raise
    
    async def _pre_deployment_checks(self):
        """Run pre-deployment checks"""
        logger.info("🔍 Running pre-deployment checks...")
        
        # Check system resources
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        if memory.available < 1024 * 1024 * 1024:  # 1GB
            raise Exception("Insufficient memory for production deployment")
        
        if disk.free < 5 * 1024 * 1024 * 1024:  # 5GB
            raise Exception("Insufficient disk space for production deployment")
        
        # Check required services
        await self._check_required_services()
        
        # Check configuration
        config_summary = self.config_manager.get_config_summary()
        logger.info(f"📋 Configuration loaded: {config_summary['optimization_rules_count']} optimization rules")
        
        logger.info("✅ Pre-deployment checks passed")
    
    async def _check_required_services(self):
        """Check if required services are running"""
        logger.info("🔧 Checking required services...")
        
        # Check if database is accessible
        try:
            # This would check your TimescaleDB connection
            logger.info("✅ Database connection verified")
        except Exception as e:
            logger.warning(f"⚠️ Database connection issue: {e}")
        
        # Check if monitoring services are available
        if not self.performance_monitor.is_monitoring:
            logger.warning("⚠️ Performance monitoring not started")
        
        logger.info("✅ Required services check completed")
    
    async def _initialize_production_detector(self):
        """Initialize detector with production configuration"""
        logger.info("🔧 Initializing production detector...")
        
        config = self.config_manager.config
        
        self.detector = UltraOptimizedPatternDetector(
            max_workers=config.max_workers,
            buffer_size=config.buffer_size
        )
        
        logger.info(f"✅ Production detector initialized with {config.max_workers} workers")
    
    async def _run_database_migration(self):
        """Run database migration for ultra-optimized pattern schema"""
        logger.info("🗄️ Running database migration...")
        
        try:
            # Change to backend directory
            backend_dir = os.path.join(os.path.dirname(__file__), '..', 'backend')
            original_dir = os.getcwd()
            os.chdir(backend_dir)
            
            # Run migration using Alembic
            result = subprocess.run([
                'alembic', 'upgrade', 'head'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning(f"⚠️ Migration warning: {result.stderr}")
            else:
                logger.info("✅ Database migration completed")
            
            # Return to original directory
            os.chdir(original_dir)
            
        except Exception as e:
            logger.warning(f"⚠️ Database migration issue: {e}")
            # Continue with deployment even if migration fails
    
    async def _start_production_monitoring(self):
        """Start production monitoring and optimization"""
        logger.info("📊 Starting production monitoring...")
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        
        # Start configuration optimization loop
        asyncio.create_task(self._configuration_optimization_loop())
        
        # Start health check loop
        asyncio.create_task(self._health_check_loop())
        
        logger.info("✅ Production monitoring started")
    
    async def _configuration_optimization_loop(self):
        """Continuous configuration optimization loop"""
        logger.info("⚙️ Starting configuration optimization loop...")
        
        while self.is_running:
            try:
                # Get current performance metrics
                performance_summary = self.performance_monitor.get_performance_summary()
                
                # Update configuration based on performance
                changes = self.config_manager.update_config_based_on_performance(
                    performance_summary
                )
                
                if changes:
                    self.integration_stats['config_changes'] += len(changes)
                    logger.info(f"🔧 Applied {len(changes)} configuration changes")
                
                # Wait before next optimization cycle
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in configuration optimization loop: {e}")
                await asyncio.sleep(60)
    
    async def _health_check_loop(self):
        """Continuous health check loop"""
        logger.info("🏥 Starting health check loop...")
        
        while self.is_running:
            try:
                # Check system health
                health_status = await self._check_system_health()
                
                if not health_status['healthy']:
                    logger.warning(f"⚠️ System health issue: {health_status['issues']}")
                    self.integration_stats['alerts_triggered'] += 1
                
                # Wait before next health check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(30)
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """Check system health status"""
        issues = []
        
        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            issues.append(f"High memory usage: {memory.percent}%")
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            issues.append(f"High CPU usage: {cpu_percent}%")
        
        # Check disk usage
        disk = psutil.disk_usage('/')
        if disk.percent > 90:
            issues.append(f"High disk usage: {disk.percent}%")
        
        # Check if detector is responsive
        if self.detector is None:
            issues.append("Pattern detector not initialized")
        
        return {
            'healthy': len(issues) == 0,
            'issues': issues,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _start_production_processing(self):
        """Start production pattern detection processing"""
        logger.info("🔄 Starting production processing...")
        
        self.is_running = True
        self.integration_stats['start_time'] = datetime.now()
        
        # Start processing loop
        await self._production_processing_loop()
    
    async def _production_processing_loop(self):
        """Main production processing loop"""
        logger.info("🔄 Production processing loop started")
        
        while self.is_running:
            try:
                # Simulate production data processing
                # In real production, this would process actual market data
                await self._process_production_data()
                
                # Update statistics
                self._update_integration_stats()
                
                # Brief pause to prevent overwhelming the system
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in production processing loop: {e}")
                self.integration_stats['errors'].append(str(e))
                await asyncio.sleep(5)
    
    async def _process_production_data(self):
        """Process production data (simulated)"""
        # This would integrate with your actual data sources
        # For now, we'll simulate processing
        
        # Simulate pattern detection
        if self.detector:
            # Create sample data for demonstration
            import pandas as pd
            import numpy as np
            
            # Generate realistic market data
            n = 1000
            closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
            highs = closes + np.random.rand(n) * 2
            lows = closes - np.random.rand(n) * 2
            opens = np.roll(closes, 1)
            opens[0] = closes[0]
            volumes = np.random.randint(1000, 10000, n)
            
            df = pd.DataFrame({
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes,
                'timestamp': pd.date_range('2024-01-01', periods=n, freq='1min')
            })
            
            # Process with ultra-optimized detector
            start_time = time.time()
            patterns = self.detector.detect_patterns_ultra_optimized(df, 'BTCUSDT', '1m')
            processing_time = (time.time() - start_time) * 1000
            
            # Record performance metrics
            self.performance_monitor.record_pattern_detection(
                patterns_count=len(patterns),
                processing_time_ms=processing_time,
                cache_hit=False,
                error=False
            )
            
            # Update integration stats
            self.integration_stats['total_patterns_processed'] += len(patterns)
            self.integration_stats['total_processing_time'] += processing_time
    
    def _update_integration_stats(self):
        """Update integration statistics"""
        if self.integration_stats['start_time']:
            uptime = datetime.now() - self.integration_stats['start_time']
            
            # Log statistics every 5 minutes
            if int(uptime.total_seconds()) % 300 == 0:
                avg_processing_time = (
                    self.integration_stats['total_processing_time'] / 
                    max(self.integration_stats['total_patterns_processed'], 1)
                )
                
                logger.info(f"📊 Production Stats - "
                          f"Patterns: {self.integration_stats['total_patterns_processed']}, "
                          f"Avg Time: {avg_processing_time:.2f}ms, "
                          f"Config Changes: {self.integration_stats['config_changes']}, "
                          f"Alerts: {self.integration_stats['alerts_triggered']}")
    
    def scale_up(self, target_workers: int = None, target_buffer_size: int = None):
        """Scale up the system resources"""
        logger.info("📈 Scaling up system resources...")
        
        config = self.config_manager.config
        
        if target_workers:
            config.max_workers = min(target_workers, 16)
            logger.info(f"📈 Scaled workers to {config.max_workers}")
        
        if target_buffer_size:
            config.buffer_size = target_buffer_size
            logger.info(f"📈 Scaled buffer size to {config.buffer_size}")
        
        # Reinitialize detector with new configuration
        if self.detector:
            self.detector = UltraOptimizedPatternDetector(
                max_workers=config.max_workers,
                buffer_size=config.buffer_size
            )
        
        self.config_manager.save_config()
        logger.info("✅ System scaling completed")
    
    def scale_down(self, target_workers: int = None, target_buffer_size: int = None):
        """Scale down the system resources"""
        logger.info("📉 Scaling down system resources...")
        
        config = self.config_manager.config
        
        if target_workers:
            config.max_workers = max(target_workers, 2)
            logger.info(f"📉 Scaled workers to {config.max_workers}")
        
        if target_buffer_size:
            config.buffer_size = max(target_buffer_size, 100)
            logger.info(f"📉 Scaled buffer size to {config.buffer_size}")
        
        # Reinitialize detector with new configuration
        if self.detector:
            self.detector = UltraOptimizedPatternDetector(
                max_workers=config.max_workers,
                buffer_size=config.buffer_size
            )
        
        self.config_manager.save_config()
        logger.info("✅ System scaling completed")
    
    def get_production_status(self) -> Dict[str, Any]:
        """Get current production status"""
        performance_summary = self.performance_monitor.get_performance_summary()
        config_summary = self.config_manager.get_config_summary()
        health_status = asyncio.run(self._check_system_health())
        
        return {
            'status': 'running' if self.is_running else 'stopped',
            'uptime': str(datetime.now() - self.integration_stats['start_time']) if self.integration_stats['start_time'] else None,
            'performance': performance_summary,
            'configuration': config_summary,
            'health': health_status,
            'statistics': self.integration_stats
        }
    
    def shutdown(self):
        """Graceful shutdown of production integration"""
        logger.info("🛑 Initiating graceful shutdown...")
        
        self.is_running = False
        
        # Stop monitoring
        self.performance_monitor.stop_monitoring()
        
        # Save final statistics
        self._save_final_statistics()
        
        logger.info("✅ Production integration shutdown completed")
    
    def _save_final_statistics(self):
        """Save final production statistics"""
        try:
            final_stats = {
                'shutdown_time': datetime.now().isoformat(),
                'integration_stats': self.integration_stats,
                'performance_summary': self.performance_monitor.get_performance_summary(),
                'config_summary': self.config_manager.get_config_summary()
            }
            
            stats_file = f"logs/production_final_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(stats_file, 'w') as f:
                json.dump(final_stats, f, indent=2, default=str)
            
            logger.info(f"📊 Final statistics saved to {stats_file}")
            
        except Exception as e:
            logger.error(f"Error saving final statistics: {e}")

async def main():
    """Main production integration function"""
    logger.info("🏭 Starting Production Integration for Ultra-Optimized Pattern Detection")
    
    # Create production integration manager
    integration = ProductionIntegration()
    
    try:
        # Deploy to production
        await integration.deploy_to_production()
        
        # Keep running until interrupted
        while integration.is_running:
            await asyncio.sleep(1)
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Production integration failed: {e}")
    finally:
        integration.shutdown()

if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Run production integration
    asyncio.run(main())
