#!/usr/bin/env python3
"""
PostgreSQL Docker Setup Script
Sets up PostgreSQL with TimescaleDB using Docker for AlphaPlus
"""

import subprocess
import sys
import time
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

def check_docker():
    """Check if Docker is installed and running"""
    logger.info("🔍 Checking Docker installation...")
    
    success, stdout, stderr = run_command("docker --version")
    if not success:
        logger.error("❌ Docker is not installed or not in PATH")
        logger.info("💡 Please install Docker Desktop from: https://www.docker.com/products/docker-desktop/")
        return False
    
    logger.info(f"✅ Docker found: {stdout.strip()}")
    
    # Check if Docker is running
    success, stdout, stderr = run_command("docker ps")
    if not success:
        logger.error("❌ Docker is not running")
        logger.info("💡 Please start Docker Desktop")
        return False
    
    logger.info("✅ Docker is running")
    return True

def setup_postgresql_docker():
    """Set up PostgreSQL with TimescaleDB using Docker"""
    logger.info("🐳 Setting up PostgreSQL with TimescaleDB using Docker...")
    
    # Stop and remove existing container if it exists
    logger.info("🔄 Stopping existing PostgreSQL container...")
    run_command("docker stop postgres-alphapulse", shell=True)
    run_command("docker rm postgres-alphapulse", shell=True)
    
    # Create PostgreSQL container with TimescaleDB
    logger.info("🔄 Creating PostgreSQL container...")
    docker_command = """
    docker run --name postgres-alphapulse 
    -e POSTGRES_PASSWORD=password 
    -e POSTGRES_DB=alphapulse 
    -e POSTGRES_USER=postgres 
    -p 5432:5432 
    -d timescale/timescaledb:latest-pg17
    """
    
    success, stdout, stderr = run_command(docker_command)
    if not success:
        logger.error(f"❌ Failed to create PostgreSQL container: {stderr}")
        return False
    
    logger.info("✅ PostgreSQL container created successfully")
    
    # Wait for PostgreSQL to start
    logger.info("⏳ Waiting for PostgreSQL to start...")
    for i in range(30):  # Wait up to 30 seconds
        success, stdout, stderr = run_command("docker exec postgres-alphapulse pg_isready -U postgres")
        if success:
            logger.info("✅ PostgreSQL is ready!")
            break
        time.sleep(1)
    else:
        logger.error("❌ PostgreSQL failed to start within 30 seconds")
        return False
    
    return True

def test_connection():
    """Test PostgreSQL connection"""
    logger.info("🔌 Testing PostgreSQL connection...")
    
    # Test connection using docker exec
    test_command = """
    docker exec postgres-alphapulse psql -U postgres -d alphapulse -c "SELECT version();"
    """
    
    success, stdout, stderr = run_command(test_command)
    if success:
        logger.info("✅ PostgreSQL connection successful!")
        logger.info(f"📊 PostgreSQL version: {stdout.strip()}")
        return True
    else:
        logger.error(f"❌ PostgreSQL connection failed: {stderr}")
        return False

def install_timescaledb():
    """Install TimescaleDB extension"""
    logger.info("🔧 Installing TimescaleDB extension...")
    
    install_command = """
    docker exec postgres-alphapulse psql -U postgres -d alphapulse -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
    """
    
    success, stdout, stderr = run_command(install_command)
    if success:
        logger.info("✅ TimescaleDB extension installed successfully!")
        return True
    else:
        logger.error(f"❌ Failed to install TimescaleDB: {stderr}")
        return False

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
    migration_command = f"""
    docker exec -i postgres-alphapulse psql -U postgres -d alphapulse << 'EOF'
    {migration_sql}
    EOF
    """
    
    success, stdout, stderr = run_command(migration_command)
    if success:
        logger.info("✅ Database migrations completed successfully!")
        return True
    else:
        logger.error(f"❌ Migration failed: {stderr}")
        return False

def main():
    """Main function"""
    logger.info("🚀 Starting PostgreSQL Docker Setup")
    logger.info("=" * 60)
    
    # Check Docker
    if not check_docker():
        logger.error("❌ Docker setup failed")
        return False
    
    # Setup PostgreSQL
    if not setup_postgresql_docker():
        logger.error("❌ PostgreSQL setup failed")
        return False
    
    # Test connection
    if not test_connection():
        logger.error("❌ Connection test failed")
        return False
    
    # Install TimescaleDB
    if not install_timescaledb():
        logger.error("❌ TimescaleDB installation failed")
        return False
    
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
    logger.info("   Password: password")
    logger.info("")
    logger.info("💡 You can now run the live signal generation test with real PostgreSQL!")
    
    return True

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
