# AlphaPulse Deployment Guide

## 🚀 Deploying to Vercel

### Prerequisites
- GitHub account
- Vercel account
- Node.js 18+ installed locally

### Step 1: Create GitHub Repository

1. **Initialize Git Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: AlphaPulse Trading Bot"
   ```

2. **Create GitHub Repository**
   - Go to [GitHub](https://github.com)
   - Click "New repository"
   - Name it `alphapulse-trading-bot`
   - Make it public or private (your choice)
   - Don't initialize with README (we already have one)

3. **Push to GitHub**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/alphapulse-trading-bot.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy to Vercel

1. **Connect to Vercel**
   - Go to [Vercel](https://vercel.com)
   - Sign in with GitHub
   - Click "New Project"
   - Import your `alphapulse-trading-bot` repository

2. **Configure Environment Variables**
   In Vercel dashboard, go to Settings → Environment Variables and add:

   ```env
   # Database
   DATABASE_URL=your_postgresql_url
   REDIS_URL=your_redis_url
   
   # API Keys
   BINANCE_API_KEY=your_binance_api_key
   BINANCE_SECRET_KEY=your_binance_secret_key
   TWITTER_API_KEY=your_twitter_api_key
   TWITTER_API_SECRET=your_twitter_api_secret
   REDDIT_CLIENT_ID=your_reddit_client_id
   REDDIT_CLIENT_SECRET=your_reddit_client_secret
   NEWS_API_KEY=your_news_api_key
   
   # Frontend
   NEXT_PUBLIC_API_URL=https://your-app.vercel.app
   NEXT_PUBLIC_WS_URL=wss://your-app.vercel.app
   ```

3. **Deploy Settings**
   - Framework Preset: `Other`
   - Build Command: `npm run build` (for frontend)
   - Output Directory: `.next`
   - Install Command: `npm install`

4. **Deploy**
   - Click "Deploy"
   - Wait for build to complete

### Step 3: Database Setup

Since Vercel doesn't support persistent databases, you'll need external services:

#### Option A: Supabase (Recommended)
1. Go to [Supabase](https://supabase.com)
2. Create new project
3. Get connection string from Settings → Database
4. Add to Vercel environment variables

#### Option B: PlanetScale
1. Go to [PlanetScale](https://planetscale.com)
2. Create new database
3. Get connection string
4. Add to Vercel environment variables

#### Option C: Railway
1. Go to [Railway](https://railway.app)
2. Create PostgreSQL service
3. Get connection string
4. Add to Vercel environment variables

### Step 4: Redis Setup

#### Option A: Upstash Redis
1. Go to [Upstash](https://upstash.com)
2. Create Redis database
3. Get connection string
4. Add to Vercel environment variables

#### Option B: Redis Cloud
1. Go to [Redis Cloud](https://redis.com/redis-enterprise-cloud/)
2. Create database
3. Get connection string
4. Add to Vercel environment variables

### Step 5: Verify Deployment

1. **Check Frontend**
   - Visit your Vercel URL
   - Should see AlphaPulse dashboard

2. **Check Backend API**
   - Visit `https://your-app.vercel.app/api/health`
   - Should return health status

3. **Check WebSocket**
   - Open browser console
   - Should see WebSocket connection status

## 🔧 Local Development

### Prerequisites
```bash
# Install Node.js 18+
# Install Python 3.9+

# Install dependencies
cd frontend && npm install
cd ../backend && pip install -r requirements.txt
```

### Environment Setup
```bash
# Create .env file
cp .env.example .env

# Edit .env with your API keys
nano .env
```

### Run Locally
```bash
# Terminal 1: Backend
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

## 📊 Monitoring

### Vercel Analytics
- Go to Vercel dashboard
- Check Analytics tab
- Monitor performance and errors

### Logs
- Vercel dashboard → Functions
- Check function logs for errors

### Database Monitoring
- Use your database provider's dashboard
- Monitor connection and performance

## 🔄 Continuous Deployment

### Automatic Deployments
- Every push to `main` branch triggers deployment
- Preview deployments for pull requests

### Environment Management
- Production: `main` branch
- Staging: `develop` branch (optional)
- Preview: Pull requests

## 🚨 Troubleshooting

### Common Issues

1. **Build Failures**
   ```bash
   # Check build logs in Vercel
   # Ensure all dependencies are in package.json
   ```

2. **Environment Variables**
   ```bash
   # Verify all env vars are set in Vercel
   # Check for typos in variable names
   ```

3. **Database Connection**
   ```bash
   # Test database connection
   # Verify connection string format
   ```

4. **API Errors**
   ```bash
   # Check Vercel function logs
   # Verify API routes are correct
   ```

### Debug Commands
```bash
# Check Vercel CLI
npm i -g vercel

# Deploy from CLI
vercel

# Check logs
vercel logs

# Pull environment variables
vercel env pull .env.local
```

## 🔐 Security

### Environment Variables
- Never commit API keys to Git
- Use Vercel environment variables
- Rotate keys regularly

### Database Security
- Use connection pooling
- Enable SSL connections
- Regular backups

### API Security
- Rate limiting
- Input validation
- CORS configuration

## 📈 Performance

### Optimization Tips
1. **Frontend**
   - Use Next.js Image component
   - Implement code splitting
   - Optimize bundle size

2. **Backend**
   - Use connection pooling
   - Implement caching
   - Optimize database queries

3. **Database**
   - Use indexes
   - Optimize queries
   - Monitor performance

## 🔄 Updates

### Deploying Updates
```bash
# Make changes
git add .
git commit -m "Update description"
git push origin main
# Vercel automatically deploys
```

### Rollback
- Go to Vercel dashboard
- Deployments tab
- Click "Redeploy" on previous deployment

## 📞 Support

### Resources
- [Vercel Documentation](https://vercel.com/docs)
- [Next.js Documentation](https://nextjs.org/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com)

### Community
- GitHub Issues
- Vercel Community
- Discord/Slack channels

---

**Happy Trading! 🚀**
