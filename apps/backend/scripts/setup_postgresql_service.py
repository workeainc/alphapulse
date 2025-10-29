#!/usr/bin/env python3
"""
PostgreSQL Service Setup Script
Sets up PostgreSQL as a Windows service
"""

import subprocess
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(command, shell=True):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=shell, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_postgresql_installation():
    """Check if PostgreSQL is installed"""
    logger.info("🔍 Checking PostgreSQL installation...")
    
    postgres_path = "C:\\Program Files\\PostgreSQL\\17"
    if not os.path.exists(postgres_path):
        logger.error("❌ PostgreSQL installation not found")
        return False
    
    logger.info("✅ PostgreSQL installation found")
    return True

def install_postgresql_service():
    """Install PostgreSQL as a Windows service"""
    logger.info("🔧 Installing PostgreSQL as Windows service...")
    
    # Try to install the service
    install_command = '"C:\\Program Files\\PostgreSQL\\17\\bin\\pg_ctl.exe" register -N "postgresql-x64-17" -D "C:\\Program Files\\PostgreSQL\\17\\data"'
    
    success, stdout, stderr = run_command(install_command)
    if success:
        logger.info("✅ PostgreSQL service installed successfully!")
        return True
    else:
        logger.error(f"❌ Failed to install PostgreSQL service: {stderr}")
        return False

def start_postgresql_service():
    """Start PostgreSQL service"""
    logger.info("🔄 Starting PostgreSQL service...")
    
    start_command = 'net start postgresql-x64-17'
    success, stdout, stderr = run_command(start_command)
    
    if success:
        logger.info("✅ PostgreSQL service started successfully!")
        return True
    else:
        logger.error(f"❌ Failed to start PostgreSQL service: {stderr}")
        return False

def test_postgresql_connection():
    """Test PostgreSQL connection"""
    logger.info("🔌 Testing PostgreSQL connection...")
    
    # Try different connection methods
    test_commands = [
        '"C:\\Program Files\\PostgreSQL\\17\\bin\\psql.exe" -U postgres -d postgres -c "SELECT version();"',
        '"C:\\Program Files\\PostgreSQL\\17\\bin\\psql.exe" -h localhost -U postgres -d postgres -c "SELECT version();"',
    ]
    
    for i, command in enumerate(test_commands, 1):
        logger.info(f"🔄 Testing connection method {i}...")
        success, stdout, stderr = run_command(command)
        
        if success:
            logger.info("✅ PostgreSQL connection successful!")
            logger.info(f"📊 PostgreSQL version: {stdout.strip()}")
            return True
        else:
            logger.warning(f"⚠️ Connection method {i} failed: {stderr}")
    
    logger.error("❌ All connection methods failed")
    return False

def create_alphapulse_database():
    """Create alphapulse database"""
    logger.info("🔧 Creating alphapulse database...")
    
    create_db_command = '"C:\\Program Files\\PostgreSQL\\17\\bin\\createdb.exe" -U postgres alphapulse'
    success, stdout, stderr = run_command(create_db_command)
    
    if success:
        logger.info("✅ alphapulse database created successfully!")
        return True
    else:
        logger.warning(f"⚠️ Database creation failed (may already exist): {stderr}")
        return True  # Continue even if database already exists

def install_timescaledb_extension():
    """Install TimescaleDB extension"""
    logger.info("🔧 Installing TimescaleDB extension...")
    
    install_command = '"C:\\Program Files\\PostgreSQL\\17\\bin\\psql.exe" -U postgres -d alphapulse -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"'
    success, stdout, stderr = run_command(install_command)
    
    if success:
        logger.info("✅ TimescaleDB extension installed successfully!")
        return True
    else:
        logger.warning(f"⚠️ TimescaleDB installation failed: {stderr}")
        logger.info("💡 TimescaleDB may not be installed. You can continue without it.")
        return True  # Continue without TimescaleDB

def run_migrations():
    """Run database migrations"""
    logger.info("🔧 Running database migrations...")
    
    # Read migration file
    try:
        with open("backend/database/migrations/create_ohlcv_hypertable.sql", "r") as f:
            migration_sql = f.read()
    except FileNotFoundError:
        logger.error("❌ Migration file not found")
        return False
    
    # Execute migration
    migration_command = f'"C:\\Program Files\\PostgreSQL\\17\\bin\\psql.exe" -U postgres -d alphapulse -c "{migration_sql}"'
    
    success, stdout, stderr = run_command(migration_command)
    if success:
        logger.info("✅ Database migrations completed successfully!")
        return True
    else:
        logger.error(f"❌ Migration failed: {stderr}")
        return False

def main():
    """Main function"""
    logger.info("🚀 Starting PostgreSQL Service Setup")
    logger.info("=" * 60)
    
    # Check PostgreSQL installation
    if not check_postgresql_installation():
        logger.error("❌ PostgreSQL installation not found")
        return False
    
    # Install service
    if not install_postgresql_service():
        logger.error("❌ Service installation failed")
        return False
    
    # Start service
    if not start_postgresql_service():
        logger.error("❌ Service start failed")
        return False
    
    # Test connection
    if not test_postgresql_connection():
        logger.error("❌ Connection test failed")
        return False
    
    # Create database
    if not create_alphapulse_database():
        logger.error("❌ Database creation failed")
        return False
    
    # Install TimescaleDB
    install_timescaledb_extension()
    
    # Run migrations
    if not run_migrations():
        logger.error("❌ Migration failed")
        return False
    
    logger.info("🎉 PostgreSQL setup completed successfully!")
    logger.info("📊 Connection details:")
    logger.info("   Host: localhost")
    logger.info("   Port: 5432")
    logger.info("   Database: alphapulse")
    logger.info("   Username: postgres")
    logger.info("   Password: [set during installation]")
    logger.info("")
    logger.info("💡 You can now run the live signal generation test with real PostgreSQL!")
    
    return True

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
