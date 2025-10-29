#!/bin/bash
# Bash Script to Apply Learning System Database Migration
# Run this after Docker is started

echo "================================================================================"
echo "🗄️  APPLYING LEARNING SYSTEM DATABASE MIGRATION"
echo "================================================================================"
echo ""

# Check if Docker is running
echo "🔍 Checking Docker status..."
if ! docker ps &> /dev/null; then
    echo "❌ Docker is not running!"
    echo ""
    echo "Please start Docker first, then run this script again."
    echo ""
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Check if PostgreSQL container is running
echo "🔍 Checking PostgreSQL container..."
if ! docker ps | grep -q alphapulse_postgres; then
    echo "❌ PostgreSQL container not found!"
    echo ""
    echo "Starting PostgreSQL container..."
    docker-compose -f ../../infrastructure/docker-compose/docker-compose.yml up -d postgres
    sleep 5
fi

echo "✅ PostgreSQL container is running"
echo ""

# Apply migration
echo "🔄 Applying learning system migration..."
echo ""

MIGRATION_PATH="src/database/migrations/003_learning_state.sql"

if [ -f "$MIGRATION_PATH" ]; then
    docker exec -i alphapulse_postgres psql -U alpha_emon -d alphapulse < "$MIGRATION_PATH"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Migration applied successfully!"
        echo ""
        
        # Verify tables created
        echo "🔍 Verifying tables created..."
        docker exec -i alphapulse_postgres psql -U alpha_emon -d alphapulse <<EOF
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN ('learning_state', 'active_learning_state', 'learning_events')
ORDER BY table_name;
EOF
        
        echo ""
        echo "================================================================================"
        echo "🎉 MIGRATION COMPLETE!"
        echo "================================================================================"
        echo ""
        echo "Next steps:"
        echo "  1. Start your backend: python main.py"
        echo "  2. Check learning system: curl http://localhost:8000/api/learning/stats"
        echo ""
        
    else
        echo ""
        echo "❌ Migration failed!"
        echo "Check the error messages above for details."
        echo ""
        exit 1
    fi
else
    echo "❌ Migration file not found: $MIGRATION_PATH"
    echo "Make sure you're in the apps/backend directory"
    echo ""
    exit 1
fi

