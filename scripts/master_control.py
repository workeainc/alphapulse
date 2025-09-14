#!/usr/bin/env python3
"""
Master Control Script for Ultra-Optimized Pattern Detection
Orchestrates monitoring, configuration, scaling, and production deployment
"""

import asyncio
import logging
import sys
import os
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
sys.path.insert(0, backend_path)

# Import services
from app.services.performance_monitor import get_performance_monitor
from app.services.config_manager import get_config_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/master_control_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class MasterControl:
    """
    Master control system for ultra-optimized pattern detection
    """
    
    def __init__(self):
        """Initialize master control"""
        self.performance_monitor = get_performance_monitor()
        self.config_manager = get_config_manager()
        self.control_stats = {
            'start_time': datetime.now(),
            'operations_performed': [],
            'config_changes': 0,
            'scaling_operations': 0,
            'alerts_handled': 0
        }
        
        logger.info("🎛️ Master Control System initialized")
    
    async def monitor_performance(self, duration_minutes: int = 60):
        """Monitor performance for specified duration"""
        logger.info(f"📊 Starting performance monitoring for {duration_minutes} minutes...")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        while time.time() < end_time:
            try:
                # Get current performance metrics
                performance_summary = self.performance_monitor.get_performance_summary()
                
                # Log performance status
                logger.info(f"📊 Performance Status - "
                          f"Patterns: {performance_summary['total_patterns_detected']}, "
                          f"Avg Time: {performance_summary['avg_processing_time_ms']:.2f}ms, "
                          f"Cache Hit Rate: {performance_summary['cache_hit_rate']:.2%}")
                
                # Check for alerts
                if performance_summary['recent_alerts']:
                    await self._handle_alerts(performance_summary['recent_alerts'])
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(30)
        
        logger.info("✅ Performance monitoring completed")
    
    async def _handle_alerts(self, alerts: List[Dict[str, Any]]):
        """Handle performance alerts"""
        for alert in alerts:
            logger.warning(f"🚨 Alert: {alert['message']}")
            
            # Record alert handling
            self.control_stats['alerts_handled'] += 1
            
            # Take action based on alert type
            if 'processing_time' in alert['metric_name']:
                await self._handle_processing_time_alert(alert)
            elif 'memory_usage' in alert['metric_name']:
                await self._handle_memory_alert(alert)
            elif 'cache_hit_rate' in alert['metric_name']:
                await self._handle_cache_alert(alert)
    
    async def _handle_processing_time_alert(self, alert: Dict[str, Any]):
        """Handle processing time alerts"""
        logger.info("⚡ Handling processing time alert...")
        
        # Increase workers if processing time is high
        current_config = self.config_manager.config
        if current_config.max_workers < 16:
            new_workers = min(current_config.max_workers + 2, 16)
            self.config_manager.config.max_workers = new_workers
            self.config_manager.save_config()
            
            logger.info(f"📈 Increased workers to {new_workers}")
            self.control_stats['config_changes'] += 1
    
    async def _handle_memory_alert(self, alert: Dict[str, Any]):
        """Handle memory usage alerts"""
        logger.info("💾 Handling memory usage alert...")
        
        # Reduce buffer size if memory usage is high
        current_config = self.config_manager.config
        if current_config.buffer_size > 500:
            new_buffer_size = max(current_config.buffer_size // 2, 500)
            self.config_manager.config.buffer_size = new_buffer_size
            self.config_manager.save_config()
            
            logger.info(f"📉 Reduced buffer size to {new_buffer_size}")
            self.control_stats['config_changes'] += 1
    
    async def _handle_cache_alert(self, alert: Dict[str, Any]):
        """Handle cache performance alerts"""
        logger.info("🗄️ Handling cache performance alert...")
        
        # Increase cache TTL if hit rate is low
        current_config = self.config_manager.config
        new_ttl = int(current_config.cache_ttl_seconds * 1.5)
        self.config_manager.config.cache_ttl_seconds = new_ttl
        self.config_manager.save_config()
        
        logger.info(f"📈 Increased cache TTL to {new_ttl} seconds")
        self.control_stats['config_changes'] += 1
    
    async def adjust_configuration(self, target_workers: int = None, target_buffer_size: int = None,
                                 target_cache_ttl: int = None, target_confidence_threshold: float = None):
        """Adjust system configuration"""
        logger.info("⚙️ Adjusting system configuration...")
        
        config = self.config_manager.config
        changes_made = []
        
        if target_workers is not None:
            old_workers = config.max_workers
            config.max_workers = max(2, min(target_workers, 16))
            if config.max_workers != old_workers:
                changes_made.append(f"Workers: {old_workers} -> {config.max_workers}")
        
        if target_buffer_size is not None:
            old_buffer = config.buffer_size
            config.buffer_size = max(100, target_buffer_size)
            if config.buffer_size != old_buffer:
                changes_made.append(f"Buffer: {old_buffer} -> {config.buffer_size}")
        
        if target_cache_ttl is not None:
            old_ttl = config.cache_ttl_seconds
            config.cache_ttl_seconds = max(60, target_cache_ttl)
            if config.cache_ttl_seconds != old_ttl:
                changes_made.append(f"Cache TTL: {old_ttl}s -> {config.cache_ttl_seconds}s")
        
        if target_confidence_threshold is not None:
            old_threshold = config.min_confidence_threshold
            config.min_confidence_threshold = max(0.1, min(target_confidence_threshold, 1.0))
            if config.min_confidence_threshold != old_threshold:
                changes_made.append(f"Confidence: {old_threshold:.3f} -> {config.min_confidence_threshold:.3f}")
        
        if changes_made:
            self.config_manager.save_config()
            self.control_stats['config_changes'] += len(changes_made)
            
            logger.info("✅ Configuration changes applied:")
            for change in changes_made:
                logger.info(f"  - {change}")
        else:
            logger.info("ℹ️ No configuration changes needed")
    
    async def scale_up(self, workers_increase: int = 2, buffer_increase_factor: float = 1.5):
        """Scale up system resources"""
        logger.info("📈 Scaling up system resources...")
        
        config = self.config_manager.config
        
        # Scale up workers
        new_workers = min(config.max_workers + workers_increase, 16)
        if new_workers != config.max_workers:
            config.max_workers = new_workers
            logger.info(f"📈 Scaled workers to {new_workers}")
        
        # Scale up buffer size
        new_buffer_size = int(config.buffer_size * buffer_increase_factor)
        if new_buffer_size != config.buffer_size:
            config.buffer_size = new_buffer_size
            logger.info(f"📈 Scaled buffer size to {new_buffer_size}")
        
        self.config_manager.save_config()
        self.control_stats['scaling_operations'] += 1
        
        logger.info("✅ System scaling up completed")
    
    async def scale_down(self, workers_decrease: int = 1, buffer_decrease_factor: float = 0.8):
        """Scale down system resources"""
        logger.info("📉 Scaling down system resources...")
        
        config = self.config_manager.config
        
        # Scale down workers
        new_workers = max(config.max_workers - workers_decrease, 2)
        if new_workers != config.max_workers:
            config.max_workers = new_workers
            logger.info(f"📉 Scaled workers to {new_workers}")
        
        # Scale down buffer size
        new_buffer_size = max(int(config.buffer_size * buffer_decrease_factor), 100)
        if new_buffer_size != config.buffer_size:
            config.buffer_size = new_buffer_size
            logger.info(f"📉 Scaled buffer size to {new_buffer_size}")
        
        self.config_manager.save_config()
        self.control_stats['scaling_operations'] += 1
        
        logger.info("✅ System scaling down completed")
    
    async def deploy_to_production(self):
        """Deploy to production environment"""
        logger.info("🏭 Deploying to production environment...")
        
        try:
            # Import and run production integration
            from scripts.production_integration import ProductionIntegration
            
            integration = ProductionIntegration()
            await integration.deploy_to_production()
            
            logger.info("✅ Production deployment completed")
            
        except Exception as e:
            logger.error(f"❌ Production deployment failed: {e}")
            raise
    
    async def run_database_migration(self):
        """Run database migration"""
        logger.info("🗄️ Running database migration...")
        
        try:
            # Import and run database migration
            from scripts.run_database_migration import DatabaseMigration
            
            migration = DatabaseMigration()
            await migration.run_complete_migration()
            
            logger.info("✅ Database migration completed")
            
        except Exception as e:
            logger.error(f"❌ Database migration failed: {e}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        performance_summary = self.performance_monitor.get_performance_summary()
        config_summary = self.config_manager.get_config_summary()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'performance': performance_summary,
            'configuration': config_summary,
            'control_stats': self.control_stats,
            'uptime': str(datetime.now() - self.control_stats['start_time'])
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive system report"""
        logger.info("📊 Generating comprehensive system report...")
        
        status = self.get_system_status()
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'system_status': status,
            'recommendations': self._generate_recommendations(status),
            'next_steps': self._generate_next_steps(status)
        }
        
        # Save report
        report_file = f"reports/master_control_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs('reports', exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 System report saved to {report_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("🎛️ MASTER CONTROL SYSTEM REPORT")
        print("="*80)
        print(f"Uptime: {status['uptime']}")
        print(f"Total Patterns: {status['performance']['total_patterns_detected']}")
        print(f"Avg Processing Time: {status['performance']['avg_processing_time_ms']:.2f}ms")
        print(f"Cache Hit Rate: {status['performance']['cache_hit_rate']:.2%}")
        print(f"Config Changes: {status['control_stats']['config_changes']}")
        print(f"Scaling Operations: {status['control_stats']['scaling_operations']}")
        print(f"Alerts Handled: {status['control_stats']['alerts_handled']}")
        print("="*80)
        
        return report_file
    
    def _generate_recommendations(self, status: Dict[str, Any]) -> List[str]:
        """Generate system recommendations"""
        recommendations = []
        
        performance = status['performance']
        config = status['configuration']['current_config']
        
        # Performance recommendations
        if performance['avg_processing_time_ms'] > 500:
            recommendations.append("High processing time - consider increasing workers or optimizing algorithms")
        
        if performance['cache_hit_rate'] < 0.4:
            recommendations.append("Low cache hit rate - consider increasing cache TTL or size")
        
        if config['max_workers'] < 8:
            recommendations.append("Low worker count - consider scaling up for better performance")
        
        if config['buffer_size'] < 500:
            recommendations.append("Small buffer size - consider increasing for better data retention")
        
        return recommendations
    
    def _generate_next_steps(self, status: Dict[str, Any]) -> List[str]:
        """Generate next steps"""
        next_steps = [
            "Continue monitoring performance metrics",
            "Adjust configuration based on performance trends",
            "Scale resources as needed for production load",
            "Run database migration when ready",
            "Deploy to production environment",
            "Set up automated alerts and notifications",
            "Implement advanced analytics and reporting",
            "Consider GPU acceleration for higher throughput"
        ]
        
        return next_steps

async def main():
    """Main control function"""
    parser = argparse.ArgumentParser(description='Master Control for Ultra-Optimized Pattern Detection')
    parser.add_argument('--action', choices=['monitor', 'configure', 'scale-up', 'scale-down', 'deploy', 'migrate', 'status', 'report'], 
                       default='status', help='Action to perform')
    parser.add_argument('--duration', type=int, default=60, help='Monitoring duration in minutes')
    parser.add_argument('--workers', type=int, help='Target number of workers')
    parser.add_argument('--buffer-size', type=int, help='Target buffer size')
    parser.add_argument('--cache-ttl', type=int, help='Target cache TTL in seconds')
    parser.add_argument('--confidence', type=float, help='Target confidence threshold')
    
    args = parser.parse_args()
    
    logger.info("🎛️ Starting Master Control for Ultra-Optimized Pattern Detection")
    
    # Create master control
    control = MasterControl()
    
    try:
        if args.action == 'monitor':
            await control.monitor_performance(args.duration)
        
        elif args.action == 'configure':
            await control.adjust_configuration(
                target_workers=args.workers,
                target_buffer_size=args.buffer_size,
                target_cache_ttl=args.cache_ttl,
                target_confidence_threshold=args.confidence
            )
        
        elif args.action == 'scale-up':
            await control.scale_up()
        
        elif args.action == 'scale-down':
            await control.scale_down()
        
        elif args.action == 'deploy':
            await control.deploy_to_production()
        
        elif args.action == 'migrate':
            await control.run_database_migration()
        
        elif args.action == 'status':
            status = control.get_system_status()
            print(json.dumps(status, indent=2, default=str))
        
        elif args.action == 'report':
            report_file = control.generate_report()
            print(f"Report generated: {report_file}")
        
    except Exception as e:
        logger.error(f"❌ Master control operation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Run master control
    asyncio.run(main())
