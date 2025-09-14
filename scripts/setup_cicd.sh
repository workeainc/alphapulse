#!/bin/bash

# 🚀 AlphaPulse CI/CD Setup Script
# This script helps you set up the CI/CD pipeline for Vercel deployment

set -e

echo "🚀 AlphaPulse CI/CD Setup Script"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_requirements() {
    print_status "Checking requirements..."
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js 18+ first."
        exit 1
    fi
    
    # Check npm
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed. Please install npm first."
        exit 1
    fi
    
    # Check Vercel CLI
    if ! command -v vercel &> /dev/null; then
        print_warning "Vercel CLI is not installed. Installing now..."
        npm install -g vercel
    fi
    
    print_success "All requirements are met!"
}

# Setup Vercel project
setup_vercel() {
    print_status "Setting up Vercel project..."
    
    if [ ! -f ".vercel/project.json" ]; then
        print_status "Linking project to Vercel..."
        vercel link --yes
    else
        print_status "Project already linked to Vercel."
    fi
    
    print_success "Vercel project setup complete!"
}

# Get Vercel credentials
get_vercel_credentials() {
    print_status "Getting Vercel credentials..."
    
    if [ -f ".vercel/project.json" ]; then
        ORG_ID=$(cat .vercel/project.json | grep -o '"orgId":"[^"]*"' | cut -d'"' -f4)
        PROJECT_ID=$(cat .vercel/project.json | grep -o '"projectId":"[^"]*"' | cut -d'"' -f4)
        
        print_success "Vercel Project ID: $PROJECT_ID"
        print_success "Vercel Org ID: $ORG_ID"
    else
        print_error "Vercel project not linked. Run 'vercel link' first."
        exit 1
    fi
    
    if [ -f "$HOME/.vercel/auth.json" ]; then
        TOKEN=$(cat $HOME/.vercel/auth.json | grep -o '"token":"[^"]*"' | cut -d'"' -f4)
        print_success "Vercel Token: $TOKEN"
    else
        print_error "Vercel token not found. Please run 'vercel login' first."
        exit 1
    fi
}

# Setup GitHub repository
setup_github() {
    print_status "Setting up GitHub repository..."
    
    if [ -d ".git" ]; then
        print_status "Git repository already exists."
    else
        print_status "Initializing Git repository..."
        git init
        git add .
        git commit -m "Initial commit: AlphaPulse Trading Bot with CI/CD"
    fi
    
    print_status "Please add your GitHub remote origin:"
    echo "git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git"
    echo "git branch -M main"
    echo "git push -u origin main"
}

# Create GitHub secrets instructions
create_github_secrets_instructions() {
    print_status "Creating GitHub secrets instructions..."
    
    cat > GITHUB_SECRETS_SETUP.md << 'EOF'
# 🔐 GitHub Secrets Setup

## Required Secrets

Add these secrets in your GitHub repository (`Settings` → `Secrets and variables` → `Actions`):

### 1. VERCEL_TOKEN
```
VERCEL_TOKEN=your_vercel_token_here
```

### 2. VERCEL_ORG_ID
```
VERCEL_ORG_ID=your_vercel_org_id_here
```

### 3. VERCEL_PROJECT_ID
```
VERCEL_PROJECT_ID=your_vercel_project_id_here
```

## How to Add Secrets

1. Go to your GitHub repository
2. Click `Settings` tab
3. Click `Secrets and variables` → `Actions`
4. Click `New repository secret`
5. Add each secret with the exact names above

## Current Values

- **VERCEL_TOKEN**: [Get from ~/.vercel/auth.json]
- **VERCEL_ORG_ID**: [Get from .vercel/project.json]
- **VERCEL_PROJECT_ID**: [Get from .vercel/project.json]

## Verification

After adding secrets, push to main branch to trigger the CI/CD pipeline:

```bash
git add .
git commit -m "Trigger CI/CD pipeline"
git push origin main
```

Check the Actions tab to see the pipeline running!
EOF

    print_success "GitHub secrets setup instructions created in GITHUB_SECRETS_SETUP.md"
}

# Setup frontend scripts
setup_frontend_scripts() {
    print_status "Setting up frontend package.json scripts..."
    
    cd frontend
    
    # Check if package.json exists
    if [ ! -f "package.json" ]; then
        print_error "frontend/package.json not found!"
        exit 1
    fi
    
    # Add required scripts if they don't exist
    if ! grep -q '"lint"' package.json; then
        print_status "Adding lint script..."
        npm pkg set scripts.lint="next lint"
    fi
    
    if ! grep -q '"type-check"' package.json; then
        print_status "Adding type-check script..."
        npm pkg set scripts.type-check="tsc --noEmit"
    fi
    
    if ! grep -q '"test:ci"' package.json; then
        print_status "Adding test:ci script..."
        npm pkg set scripts.test:ci="jest --ci --coverage --watchAll=false"
    fi
    
    if ! grep -q '"build"' package.json; then
        print_status "Adding build script..."
        npm pkg set scripts.build="next build"
    fi
    
    cd ..
    print_success "Frontend scripts configured!"
}

# Setup backend testing dependencies
setup_backend_dependencies() {
    print_status "Setting up backend testing dependencies..."
    
    cd backend
    
    # Check if requirements.txt exists
    if [ ! -f "requirements.txt" ]; then
        print_error "backend/requirements.txt not found!"
        exit 1
    fi
    
    # Add testing dependencies if they don't exist
    TEST_DEPS=(
        "pytest>=7.0.0"
        "pytest-cov>=4.0.0"
        "pytest-asyncio>=0.21.0"
        "black>=23.0.0"
        "flake8>=6.0.0"
        "mypy>=1.0.0"
    )
    
    for dep in "${TEST_DEPS[@]}"; do
        if ! grep -q "$(echo $dep | cut -d'>' -f1)" requirements.txt; then
            print_status "Adding $dep..."
            echo "$dep" >> requirements.txt
        fi
    done
    
    cd ..
    print_success "Backend testing dependencies configured!"
}

# Test the setup
test_setup() {
    print_status "Testing the setup..."
    
    # Test frontend build
    cd frontend
    print_status "Testing frontend build..."
    npm run build
    cd ..
    
    # Test backend imports
    cd backend
    print_status "Testing backend imports..."
    python -c "import fastapi; print('FastAPI import successful')"
    cd ..
    
    print_success "Setup test completed successfully!"
}

# Main execution
main() {
    echo ""
    print_status "Starting AlphaPulse CI/CD setup..."
    echo ""
    
    check_requirements
    echo ""
    
    setup_vercel
    echo ""
    
    get_vercel_credentials
    echo ""
    
    setup_frontend_scripts
    echo ""
    
    setup_backend_dependencies
    echo ""
    
    setup_github
    echo ""
    
    create_github_secrets_instructions
    echo ""
    
    test_setup
    echo ""
    
    print_success "🎉 AlphaPulse CI/CD setup completed successfully!"
    echo ""
    print_status "Next steps:"
    echo "1. Add the GitHub secrets as shown in GITHUB_SECRETS_SETUP.md"
    echo "2. Push your code to GitHub main branch"
    echo "3. Watch the CI/CD pipeline run in the Actions tab"
    echo ""
    print_status "Happy trading! 🚀"
}

# Run main function
main "$@"
