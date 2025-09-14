# 🚀 CI/CD Pipeline Documentation

## Overview

AlphaPulse uses GitHub Actions for continuous integration and deployment to Vercel. The pipeline automatically tests, builds, and deploys your application whenever code is pushed to the main branch.

## 🏗️ Pipeline Architecture

### Workflow Triggers
- **Push to main**: Triggers full CI/CD pipeline
- **Push to develop**: Triggers testing only (no deployment)
- **Pull Request to main**: Triggers testing only
- **Manual trigger**: Available via GitHub Actions UI

### Pipeline Stages

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Code Push    │───▶│   Backend Test  │───▶│   Frontend Test │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Security Scan   │    │   Build Check   │
                       └─────────────────┘    └─────────────────┘
                                │                       │
                                └───────────┬───────────┘
                                            ▼
                                   ┌─────────────────┐
                                   │ Vercel Deploy  │
                                   └─────────────────┘
                                            │
                                            ▼
                                   ┌─────────────────┐
                                   │   Notify Team  │
                                   └─────────────────┘
```

## 🔧 Setup Requirements

### 1. GitHub Repository Secrets

Add these secrets in your GitHub repository (`Settings` → `Secrets and variables` → `Actions`):

```bash
VERCEL_TOKEN=your_vercel_token_here
VERCEL_ORG_ID=your_vercel_org_id_here
VERCEL_PROJECT_ID=your_vercel_project_id_here
```

#### How to Get Vercel Credentials:

1. **VERCEL_TOKEN**:
   ```bash
   # Install Vercel CLI globally
   npm i -g vercel
   
   # Login to Vercel
   vercel login
   
   # Get your token from ~/.vercel/auth.json
   cat ~/.vercel/auth.json
   ```

2. **VERCEL_ORG_ID** and **VERCEL_PROJECT_ID**:
   ```bash
   # In your project directory
   vercel link
   
   # This will create .vercel/project.json with your IDs
   cat .vercel/project.json
   ```

### 2. Frontend Package.json Scripts

Ensure your `frontend/package.json` has these scripts:

```json
{
  "scripts": {
    "lint": "next lint",
    "type-check": "tsc --noEmit",
    "test:ci": "jest --ci --coverage --watchAll=false",
    "build": "next build"
  }
}
```

### 3. Backend Testing Dependencies

Add these to your `backend/requirements.txt`:

```txt
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-asyncio>=0.21.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.0.0
```

## 📋 Pipeline Jobs

### 1. Backend Testing (`backend-test`)

**Purpose**: Test Python backend code quality and functionality

**Services**:
- TimescaleDB (PostgreSQL) for testing database operations

**Steps**:
- ✅ Code checkout
- 🐍 Python 3.9 setup
- 📦 Dependency installation
- 🔍 Code linting (Black, Flake8)
- 🔍 Type checking (MyPy)
- 🧪 Unit testing with coverage
- 📊 Coverage reporting

**Outputs**:
- Test results
- Coverage reports
- Linting results

### 2. Frontend Testing (`frontend-test`)

**Purpose**: Test Next.js frontend code quality and functionality

**Steps**:
- ✅ Code checkout
- 🟢 Node.js 18 setup
- 📦 NPM dependency installation
- 🔍 ESLint linting
- 🔍 TypeScript type checking
- 🧪 Jest unit testing
- 🏗️ Production build verification

**Outputs**:
- Test results
- Build artifacts
- Linting results

### 3. Security Scanning (`security-scan`)

**Purpose**: Identify security vulnerabilities in dependencies and code

**Tools**:
- **Trivy**: Vulnerability scanner for containers and dependencies
- **CodeQL**: GitHub's semantic code analysis engine

**Outputs**:
- Security scan results
- Vulnerability reports
- GitHub Security tab integration

### 4. Deployment (`deploy`)

**Purpose**: Deploy to Vercel production environment

**Triggers**: Only on successful completion of all previous jobs

**Steps**:
- ✅ Code checkout
- 🚀 Vercel CLI setup
- 🌐 Production deployment

**Outputs**:
- Live application URL
- Deployment status

### 5. Notifications (`notify`)

**Purpose**: Inform team of deployment results

**Triggers**: Always runs after deployment (success or failure)

**Outputs**:
- Success/failure messages
- Ready for Discord/Telegram webhook integration

## 🚀 Deployment Process

### Automatic Deployment
1. **Code Push**: Developer pushes to `main` branch
2. **Pipeline Trigger**: GitHub Actions automatically starts
3. **Testing**: All tests must pass
4. **Security Check**: Vulnerability scan completes
5. **Deployment**: Automatic deployment to Vercel
6. **Notification**: Team gets deployment status

### Manual Deployment
1. Go to GitHub Actions tab
2. Select "CI/CD Pipeline" workflow
3. Click "Run workflow"
4. Select branch and click "Run workflow"

## 📊 Monitoring & Debugging

### Pipeline Status
- **Green**: All tests passed, deployment successful
- **Yellow**: Tests passed, deployment in progress
- **Red**: Tests failed or deployment failed

### Common Issues & Solutions

#### 1. Backend Tests Failing
```bash
# Local testing
cd backend
pytest --cov=. --cov-report=html
```

#### 2. Frontend Build Failing
```bash
# Local testing
cd frontend
npm run build
```

#### 3. Vercel Deployment Issues
```bash
# Check Vercel status
vercel status

# Check project configuration
vercel project ls
```

### Logs & Artifacts
- **Pipeline logs**: Available in GitHub Actions UI
- **Test coverage**: Generated as HTML reports
- **Build artifacts**: Available for download
- **Security reports**: Integrated with GitHub Security tab

## 🔒 Security Features

### Vulnerability Scanning
- **Dependency scanning**: Checks npm and pip packages
- **Code scanning**: Identifies potential security issues
- **Container scanning**: Checks Docker images (if used)

### Access Control
- **Secrets management**: Sensitive data stored in GitHub Secrets
- **Branch protection**: Only main branch triggers deployment
- **Required checks**: All tests must pass before deployment

## 📈 Performance Optimization

### Caching Strategies
- **Python dependencies**: Cached between runs
- **Node modules**: NPM cache optimization
- **Build artifacts**: Reused when possible

### Parallel Execution
- Backend and frontend tests run in parallel
- Security scanning runs independently
- Deployment waits for all tests to complete

## 🛠️ Customization

### Environment-Specific Deployments
```yaml
# Add to workflow for staging deployment
- name: Deploy to Staging
  if: github.ref == 'refs/heads/develop'
  run: vercel --token ${{ secrets.VERCEL_TOKEN }}
```

### Additional Notifications
```yaml
# Add Discord webhook
- name: Discord Notification
  run: |
    curl -H "Content-Type: application/json" \
         -d '{"content":"🚀 AlphaPulse deployed successfully!"}' \
         ${{ secrets.DISCORD_WEBHOOK }}
```

### Custom Test Commands
```yaml
# Add integration tests
- name: Run Integration Tests
  run: |
    cd backend
    pytest tests/integration/ --cov=.
```

## 📚 Best Practices

### 1. Code Quality
- Write comprehensive tests
- Use type hints (Python) and TypeScript
- Follow linting rules
- Keep dependencies updated

### 2. Security
- Never commit secrets
- Use dependency scanning
- Regular security audits
- Follow OWASP guidelines

### 3. Performance
- Optimize build times
- Use caching effectively
- Parallel job execution
- Monitor pipeline metrics

### 4. Monitoring
- Set up alerts for failures
- Monitor deployment frequency
- Track test coverage trends
- Review security scan results

## 🔄 Rollback Procedures

### Automatic Rollback
- Vercel automatically rolls back on build failures
- Previous successful deployment remains active

### Manual Rollback
```bash
# Rollback to previous deployment
vercel rollback

# Or via Vercel dashboard
# Deployments → Select version → Promote to Production
```

## 📞 Support & Troubleshooting

### Common Commands
```bash
# Check workflow status
gh run list --workflow="CI/CD Pipeline"

# View workflow logs
gh run view --log

# Rerun failed workflow
gh run rerun <run-id>
```

### Useful Links
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Vercel CLI Documentation](https://vercel.com/docs/cli)
- [AlphaPulse Project](https://github.com/easoftlab/alphapulse-trading-bot)

---

**Last Updated**: $(date)
**Pipeline Version**: 1.0.0
**Maintainer**: AlphaPulse Team
