#!/usr/bin/env python3
"""
AlphaPulse Trading Bot - Unified Main Entry Point
Consolidates all main functionality into a single, comprehensive entry point
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import argparse

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Environment variables loaded from .env file")
except ImportError as e:
    print(f"python-dotenv not available: {e}, using system environment variables")

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import the unified configuration
try:
    from app.core.unified_config import get_settings, get_logger
    settings = get_settings()
    logger = get_logger(__name__)
    print("Using unified configuration")
except ImportError as e:
    print(f"Unified configuration not available: {e}")
    # Fallback configuration
    class FallbackSettings:
        DEBUG = os.getenv("DEBUG", "false").lower() == "true"
        LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        APP_NAME = "AlphaPulse Trading Bot"
        VERSION = "1.0.0"
        DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/alphapulse")
        REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    settings = FallbackSettings()
    
    # Fallback logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('alphapulse.log')
        ]
    )
    logger = logging.getLogger(__name__)
    print("Using fallback configuration")

class AlphaPulseMain:
    """Unified main application class for AlphaPulse"""
    
    def __init__(self):
        self.settings = settings
        self.logger = logger
        self.running = False
        
    def print_banner(self):
        """Print the AlphaPulse banner"""
        print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    AlphaPulse Trading Bot                    ║
    ║                                                              ║
    ║  Advanced Multi-Strategy Trading System with Real-time      ║
    ║  Dashboard and Adaptive Market Detection                     ║
    ║                                                              ║
    ║  Version: {:<47} ║
    ╚══════════════════════════════════════════════════════════════╝
        """.format(getattr(self.settings, 'VERSION', '1.0.0')))
    
    def check_requirements(self) -> bool:
        """Check if required software is installed"""
        self.logger.info("Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.logger.error("Python 3.8 or higher is required")
            return False
        
        # Check if we can import key modules
        required_modules = [
            'sqlalchemy',
            'pandas',
            'numpy',
            'asyncio',
            'psutil'
        ]
        
        for module in required_modules:
            try:
                __import__(module)
                self.logger.info(f"{module} is available")
            except ImportError:
                self.logger.warning(f"{module} is not available")
        
        return True
    
    async def initialize_database(self) -> bool:
        """Initialize database connection with retry logic and specific error handling"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Initializing database connection (attempt {attempt + 1}/{max_retries})...")
                
                # Try to import and initialize database connection
                try:
                    from ..database.enhanced_connection import EnhancedTimescaleDBConnection
                    connection = EnhancedTimescaleDBConnection()
                    await connection.initialize()
                    
                    # Test health check
                    health = await connection.health_check()
                    self.logger.info(f"Database health check: {health}")
                    
                    await connection.close()
                    return True
                    
                except ImportError as e:
                    self.logger.warning(f"Enhanced connection not available: {e}, trying basic connection...")
                    try:
                        from ..database.connection import TimescaleDBConnection
                        connection = TimescaleDBConnection()
                        connection.initialize()
                        
                        # Test health check
                        health = await connection.health_check()
                        self.logger.info(f"Database health check: {health}")
                        
                        await connection.close()
                        return True
                        
                    except ImportError as import_err:
                        self.logger.error(f"Database connection module not found: {import_err}", exc_info=True)
                        return False
                    except ConnectionError as conn_err:
                        self.logger.error(f"Database connection failed: {conn_err}", exc_info=True)
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            continue
                        return False
                    except Exception as e:
                        self.logger.error(f"Basic database connection failed: {e}", exc_info=True)
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            continue
                        return False
                        
            except ConnectionError as e:
                self.logger.error(f"Database connection error (attempt {attempt + 1}): {e}", exc_info=True)
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                return False
            except Exception as e:
                self.logger.error(f"Database initialization failed (attempt {attempt + 1}): {e}", exc_info=True)
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                return False
        
        return False
    
    async def initialize_performance_framework(self) -> bool:
        """Initialize performance testing framework"""
        try:
            self.logger.info("Initializing performance testing framework...")
            
            # Import performance framework components
            try:
                from app.core.performance_profiling import get_performance_profiler
                from app.core.benchmark_framework import get_benchmark_framework
                from app.core.performance_regression import get_performance_regression_tester
                
                # Initialize components
                profiler = get_performance_profiler()
                benchmark_framework = get_benchmark_framework()
                regression_tester = get_performance_regression_tester()
                
                self.logger.info("Performance testing framework initialized")
                return True
                
            except ImportError as e:
                self.logger.warning(f"Performance framework not available: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Performance framework initialization failed: {e}")
            return False
    
    async def start_dashboards(self) -> bool:
        """Start monitoring dashboards"""
        try:
            self.logger.info("Starting monitoring dashboards...")
            
            # Try to start various dashboards
            dashboards = [
                ("Resilience Dashboard", "run_resilience_dashboard.py"),
                ("Analytics Dashboard", "run_analytics_dashboard.py"),
                ("Security Dashboard", "run_security_dashboard.py"),
                ("Multi-Region Dashboard", "run_multi_region_dashboard.py"),
                ("Chaos Engineering", "run_chaos_engineering.py")
            ]
            
            started_count = 0
            for name, script in dashboards:
                try:
                    script_path = Path(script)
                    if script_path.exists():
                        self.logger.info(f"{name} script found")
                        started_count += 1
                    else:
                        self.logger.warning(f"{name} script not found: {script}")
                except Exception as e:
                    self.logger.warning(f"Error checking {name}: {e}")
            
            self.logger.info(f"Available {started_count}/{len(dashboards)} dashboard scripts")
            return started_count > 0
            
        except Exception as e:
            self.logger.error(f"Dashboard startup failed: {e}")
            return False
    
    async def run_performance_test(self) -> bool:
        """Run a quick performance test"""
        try:
            self.logger.info("Running quick performance test...")
            
            # Import and run basic performance test
            try:
                from app.core.performance_profiling import get_performance_profiler
                
                profiler = get_performance_profiler()
                
                # Simple test function
                def test_function():
                    import time
                    time.sleep(0.1)
                    return [i * 2 for i in range(1000)]
                
                # Profile the function
                @profiler.profile_function(output_file="startup_performance_test")
                def profiled_test():
                    return test_function()
                
                result = profiled_test()
                
                if result and len(result) == 1000:
                    self.logger.info("Performance test passed")
                    return True
                else:
                    self.logger.warning("Performance test failed")
                    return False
                    
            except ImportError as e:
                self.logger.warning(f"Performance testing not available: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Performance test failed: {e}")
            return False
    
    async def main_loop(self):
        """Main application loop"""
        try:
            self.logger.info("Starting AlphaPulse main loop...")
            self.logger.info("Press Ctrl+C to stop the application")
            
            self.running = True
            iteration = 0
            
            while self.running:
                iteration += 1
                self.logger.info(f"Main loop iteration {iteration}")
                
                # Check system health
                try:
                    import psutil
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    
                    self.logger.info(f"System Status - CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%")
                    
                except ImportError:
                    self.logger.warning("psutil not available for system monitoring")
                
                # Wait before next iteration
                await asyncio.sleep(30)  # 30 seconds between iterations
                
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, shutting down...")
            self.running = False
        except Exception as e:
            self.logger.error(f"Main loop error: {e}")
            self.running = False
    
    async def run(self, mode: str = "full"):
        """Run the main application"""
        try:
            self.print_banner()
            
            # Check requirements
            if not self.check_requirements():
                self.logger.error("System requirements not met")
                return False
            
            # Initialize components based on mode
            if mode in ["full", "database"]:
                if not await self.initialize_database():
                    self.logger.warning("Database initialization failed, continuing...")
            
            if mode in ["full", "performance"]:
                if not await self.initialize_performance_framework():
                    self.logger.warning("Performance framework initialization failed, continuing...")
            
            if mode in ["full", "dashboards"]:
                if not await self.start_dashboards():
                    self.logger.warning("Dashboard startup failed, continuing...")
            
            if mode in ["full", "performance"]:
                if not await self.run_performance_test():
                    self.logger.warning("Performance test failed, continuing...")
            
            # Start main loop
            if mode == "full":
                await self.main_loop()
            
            self.logger.info("AlphaPulse startup completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"AlphaPulse startup failed: {e}")
            return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AlphaPulse Trading Bot")
    parser.add_argument(
        "--mode", 
        choices=["full", "database", "performance", "dashboards", "test"],
        default="full",
        help="Run mode (default: full)"
    )
    
    args = parser.parse_args()
    
    # Create and run main application
    app = AlphaPulseMain()
    
    try:
        if args.mode == "test":
            # Run in test mode (no main loop)
            success = asyncio.run(app.run("database"))
        else:
            # Run in specified mode
            success = asyncio.run(app.run(args.mode))
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nApplication failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
