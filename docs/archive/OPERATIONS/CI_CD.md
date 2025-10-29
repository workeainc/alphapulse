# 🚀 AlphaPulse CI/CD Setup Guide

This guide will help you set up the complete CI/CD pipeline for AlphaPulse Trading Bot, enabling automatic testing, building, and deployment to Vercel.

## 🎯 What You'll Get

- ✅ **Automated Testing**: Backend (Python) and Frontend (Next.js) testing
- 🔒 **Security Scanning**: Vulnerability detection and code quality checks
- 🚀 **Automatic Deployment**: Deploy to Vercel on every push to main branch
- 📊 **Coverage Reports**: Test coverage tracking and reporting
- 🔔 **Notifications**: Deployment status updates

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

#### For Windows Users:
```powershell
# Run the PowerShell setup script
.\scripts\setup_cicd.ps1
```

#### For Linux/Mac Users:
```bash
# Make the script executable and run
chmod +x scripts/setup_cicd.sh
./scripts/setup_cicd.sh
```

### Option 2: Manual Setup

Follow the step-by-step instructions below.

## 📋 Prerequisites

- [Node.js 18+](https://nodejs.org/)
- [Python 3.9+](https://python.org/)
- [Git](https://git-scm.com/)
- [GitHub Account](https://github.com/)
- [Vercel Account](https://vercel.com/)

## 🔧 Step-by-Step Setup

### 1. Install Vercel CLI

```bash
npm install -g vercel
```

### 2. Login to Vercel

```bash
vercel login
```

### 3. Link Your Project

```bash
vercel link
```

This will create a `.vercel/project.json` file with your project configuration.

### 4. Get Your Credentials

#### Vercel Token
```bash
# On Linux/Mac
cat ~/.vercel/auth.json

# On Windows
type %USERPROFILE%\.vercel\auth.json
```

#### Project and Org IDs
```bash
cat .vercel/project.json
```

### 5. Add GitHub Secrets

1. Go to your GitHub repository
2. Click `Settings` → `Secrets and variables` → `Actions`
3. Add these secrets:

| Secret Name | Value |
|-------------|-------|
| `VERCEL_TOKEN` | Your Vercel token from step 4 |
| `VERCEL_ORG_ID` | Your Vercel org ID from step 4 |
| `VERCEL_PROJECT_ID` | Your Vercel project ID from step 4 |

### 6. Push to GitHub

```bash
git add .
git commit -m "Add CI/CD pipeline"
git push origin main
```

## 🏗️ Pipeline Overview

The CI/CD pipeline consists of 5 main jobs:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Backend Test  │    │  Frontend Test  │    │ Security Scan   │
│   (Python)      │    │   (Next.js)     │    │  (Trivy)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ Vercel Deploy   │
                        │   (Production)  │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   Notify Team   │
                        └─────────────────┘
```

## 📊 What Gets Tested

### Backend Testing
- ✅ Code linting (Black, Flake8)
- ✅ Type checking (MyPy)
- ✅ Unit tests with coverage
- ✅ Database connectivity (TimescaleDB)

### Frontend Testing
- ✅ Code linting (ESLint)
- ✅ TypeScript type checking
- ✅ Unit tests (Jest)
- ✅ Production build verification

### Security Scanning
- ✅ Dependency vulnerability scanning
- ✅ Code security analysis
- ✅ GitHub Security tab integration

## 🔍 Monitoring Your Pipeline

### GitHub Actions Tab
1. Go to your repository
2. Click `Actions` tab
3. Select `CI/CD Pipeline` workflow
4. Monitor job progress and logs

### Pipeline Status
- 🟢 **Green**: All tests passed, deployment successful
- 🟡 **Yellow**: Tests passed, deployment in progress
- 🔴 **Red**: Tests failed or deployment failed

## 🛠️ Customization

### Environment-Specific Deployments

Add this to your workflow for staging deployment:

```yaml
- name: Deploy to Staging
  if: github.ref == 'refs/heads/develop'
  run: vercel --token ${{ secrets.VERCEL_TOKEN }}
```

### Custom Notifications

Add Discord webhook notifications:

```yaml
- name: Discord Notification
  run: |
    curl -H "Content-Type: application/json" \
         -d '{"content":"🚀 AlphaPulse deployed successfully!"}' \
         ${{ secrets.DISCORD_WEBHOOK }}
```

### Custom Test Commands

Add integration tests:

```yaml
- name: Run Integration Tests
  run: |
    cd backend
    pytest tests/integration/ --cov=.
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Backend Tests Failing
```bash
cd backend
pytest --cov=. --cov-report=html
```

#### 2. Frontend Build Failing
```bash
cd frontend
npm run build
```

#### 3. Vercel Deployment Issues
```bash
vercel status
vercel project ls
```

#### 4. GitHub Actions Not Running
- Check repository permissions
- Verify workflow file exists in `.github/workflows/`
- Ensure secrets are properly configured

### Debug Commands

```bash
# Check workflow status
gh run list --workflow="CI/CD Pipeline"

# View workflow logs
gh run view --log

# Rerun failed workflow
gh run rerun <run-id>
```

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Vercel CLI Documentation](https://vercel.com/docs/cli)
- [Jest Testing Framework](https://jestjs.io/)
- [Pytest Testing Framework](https://pytest.org/)
- [AlphaPulse Project](https://github.com/easoftlab/alphapulse-trading-bot)

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

## 📈 Performance Tips

### Optimize Build Times
- Use dependency caching
- Run tests in parallel
- Optimize Docker images (if using)

### Monitor Metrics
- Track pipeline execution time
- Monitor test coverage trends
- Review security scan results

## 🎉 Success Checklist

- [ ] Vercel CLI installed and logged in
- [ ] Project linked to Vercel
- [ ] GitHub secrets configured
- [ ] Code pushed to main branch
- [ ] Pipeline running successfully
- [ ] First deployment completed
- [ ] Notifications working

## 🆘 Need Help?

If you encounter issues:

1. Check the troubleshooting section above
2. Review GitHub Actions logs
3. Verify all secrets are configured
4. Check Vercel project status
5. Open an issue in the repository

---

**Happy Trading! 🚀**

*This CI/CD pipeline will automatically deploy your AlphaPulse Trading Bot to Vercel on every push to the main branch, ensuring your trading system is always up-to-date and thoroughly tested.*
