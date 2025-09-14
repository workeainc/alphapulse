# 🚀 AlphaPulse CI/CD Setup Script (PowerShell)
# This script helps you set up the CI/CD pipeline for Vercel deployment

param(
    [switch]$SkipTests
)

# Error handling
$ErrorActionPreference = "Stop"

Write-Host "🚀 AlphaPulse CI/CD Setup Script" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check if required tools are installed
function Test-Requirements {
    Write-Status "Checking requirements..."
    
    # Check Node.js
    try {
        $nodeVersion = node --version 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "Node.js not found"
        }
        Write-Success "Node.js version: $nodeVersion"
    }
    catch {
        Write-Error "Node.js is not installed. Please install Node.js 18+ first."
        exit 1
    }
    
    # Check npm
    try {
        $npmVersion = npm --version 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "npm not found"
        }
        Write-Success "npm version: $npmVersion"
    }
    catch {
        Write-Error "npm is not installed. Please install npm first."
        exit 1
    }
    
    # Check Vercel CLI
    try {
        $vercelVersion = vercel --version 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Vercel CLI is not installed. Installing now..."
            npm install -g vercel
        } else {
            Write-Success "Vercel CLI version: $vercelVersion"
        }
    }
    catch {
        Write-Error "Failed to install Vercel CLI"
        exit 1
    }
    
    Write-Success "All requirements are met!"
}

# Setup Vercel project
function Setup-Vercel {
    Write-Status "Setting up Vercel project..."
    
    if (-not (Test-Path ".vercel/project.json")) {
        Write-Status "Linking project to Vercel..."
        vercel link --yes
    } else {
        Write-Status "Project already linked to Vercel."
    }
    
    Write-Success "Vercel project setup complete!"
}

# Get Vercel credentials
function Get-VercelCredentials {
    Write-Status "Getting Vercel credentials..."
    
    if (Test-Path ".vercel/project.json") {
        $projectJson = Get-Content ".vercel/project.json" | ConvertFrom-Json
        $orgId = $projectJson.orgId
        $projectId = $projectJson.projectId
        
        Write-Success "Vercel Project ID: $projectId"
        Write-Success "Vercel Org ID: $orgId"
    } else {
        Write-Error "Vercel project not linked. Run 'vercel link' first."
        exit 1
    }
    
    $vercelAuthPath = "$env:USERPROFILE\.vercel\auth.json"
    if (Test-Path $vercelAuthPath) {
        $authJson = Get-Content $vercelAuthPath | ConvertFrom-Json
        $token = $authJson.token
        Write-Success "Vercel Token: $token"
    } else {
        Write-Error "Vercel token not found. Please run 'vercel login' first."
        exit 1
    }
}

# Setup GitHub repository
function Setup-GitHub {
    Write-Status "Setting up GitHub repository..."
    
    if (Test-Path ".git") {
        Write-Status "Git repository already exists."
    } else {
        Write-Status "Initializing Git repository..."
        git init
        git add .
        git commit -m "Initial commit: AlphaPulse Trading Bot with CI/CD"
    }
    
    Write-Status "Please add your GitHub remote origin:"
    Write-Host "git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git" -ForegroundColor Yellow
    Write-Host "git branch -M main" -ForegroundColor Yellow
    Write-Host "git push -u origin main" -ForegroundColor Yellow
}

# Create GitHub secrets instructions
function Create-GitHubSecretsInstructions {
    Write-Status "Creating GitHub secrets instructions..."
    
    $instructions = @"
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
"@

    $instructions | Out-File -FilePath "GITHUB_SECRETS_SETUP.md" -Encoding UTF8
    Write-Success "GitHub secrets setup instructions created in GITHUB_SECRETS_SETUP.md"
}

# Setup frontend scripts
function Setup-FrontendScripts {
    Write-Status "Setting up frontend package.json scripts..."
    
    Push-Location frontend
    
    # Check if package.json exists
    if (-not (Test-Path "package.json")) {
        Write-Error "frontend/package.json not found!"
        exit 1
    }
    
    # Add required scripts if they don't exist
    $packageJson = Get-Content "package.json" | ConvertFrom-Json
    
    if (-not $packageJson.scripts.PSObject.Properties.Name.Contains("lint")) {
        Write-Status "Adding lint script..."
        $packageJson.scripts | Add-Member -Name "lint" -Value "next lint" -MemberType NoteProperty
    }
    
    if (-not $packageJson.scripts.PSObject.Properties.Name.Contains("type-check")) {
        Write-Status "Adding type-check script..."
        $packageJson.scripts | Add-Member -Name "type-check" -Value "tsc --noEmit" -MemberType NoteProperty
    }
    
    if (-not $packageJson.scripts.PSObject.Properties.Name.Contains("test:ci")) {
        Write-Status "Adding test:ci script..."
        $packageJson.scripts | Add-Member -Name "test:ci" -Value "jest --ci --coverage --watchAll=false" -MemberType NoteProperty
    }
    
    if (-not $packageJson.scripts.PSObject.Properties.Name.Contains("build")) {
        Write-Status "Adding build script..."
        $packageJson.scripts | Add-Member -Name "build" -Value "next build" -MemberType NoteProperty
    }
    
    # Save updated package.json
    $packageJson | ConvertTo-Json -Depth 10 | Out-File -FilePath "package.json" -Encoding UTF8
    
    Pop-Location
    Write-Success "Frontend scripts configured!"
}

# Setup backend testing dependencies
function Setup-BackendDependencies {
    Write-Status "Setting up backend testing dependencies..."
    
    Push-Location backend
    
    # Check if requirements.txt exists
    if (-not (Test-Path "requirements.txt")) {
        Write-Error "backend/requirements.txt not found!"
        exit 1
    }
    
    # Add testing dependencies if they don't exist
    $testDeps = @(
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-asyncio>=0.21.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0"
    )
    
    foreach ($dep in $testDeps) {
        $depName = $dep.Split('>')[0]
        if (-not (Select-String -Path "requirements.txt" -Pattern $depName -Quiet)) {
            Write-Status "Adding $dep..."
            Add-Content -Path "requirements.txt" -Value $dep
        }
    }
    
    Pop-Location
    Write-Success "Backend testing dependencies configured!"
}

# Test the setup
function Test-Setup {
    if ($SkipTests) {
        Write-Warning "Skipping setup tests as requested."
        return
    }
    
    Write-Status "Testing the setup..."
    
    # Test frontend build
    Push-Location frontend
    Write-Status "Testing frontend build..."
    npm run build
    Pop-Location
    
    # Test backend imports
    Push-Location backend
    Write-Status "Testing backend imports..."
    python -c "import fastapi; print('FastAPI import successful')"
    Pop-Location
    
    Write-Success "Setup test completed successfully!"
}

# Main execution
function Main {
    Write-Host ""
    Write-Status "Starting AlphaPulse CI/CD setup..."
    Write-Host ""
    
    Test-Requirements
    Write-Host ""
    
    Setup-Vercel
    Write-Host ""
    
    Get-VercelCredentials
    Write-Host ""
    
    Setup-FrontendScripts
    Write-Host ""
    
    Setup-BackendDependencies
    Write-Host ""
    
    Setup-GitHub
    Write-Host ""
    
    Create-GitHubSecretsInstructions
    Write-Host ""
    
    Test-Setup
    Write-Host ""
    
    Write-Success "🎉 AlphaPulse CI/CD setup completed successfully!"
    Write-Host ""
    Write-Status "Next steps:"
    Write-Host "1. Add the GitHub secrets as shown in GITHUB_SECRETS_SETUP.md" -ForegroundColor Yellow
    Write-Host "2. Push your code to GitHub main branch" -ForegroundColor Yellow
    Write-Host "3. Watch the CI/CD pipeline run in the Actions tab" -ForegroundColor Yellow
    Write-Host ""
    Write-Status "Happy trading! 🚀"
}

# Run main function
Main
