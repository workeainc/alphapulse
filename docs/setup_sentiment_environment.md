# Enhanced Sentiment Analysis Environment Setup Guide

## ✅ Step 1: Database Migration - COMPLETED

The database migration has been successfully completed! The following tables were created:

- `enhanced_sentiment_data` - Main sentiment storage
- `real_time_sentiment_aggregation` - Aggregated sentiment data
- `sentiment_correlation` - Price-sentiment correlation
- `sentiment_alerts` - Sentiment alerts
- `sentiment_model_performance` - Model performance tracking

All tables include TimescaleDB hypertables and optimized indexes.

## ✅ Step 2: Dependencies Installation - COMPLETED

All required Python packages have been successfully installed:

- `transformers` - Advanced NLP models
- `torch` - PyTorch for deep learning
- `textblob` - Simple NLP library
- `vaderSentiment` - VADER sentiment analysis
- `redis` - Redis client for caching
- `aiohttp` - Async HTTP client
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `websockets` - WebSocket support

## 🔧 Step 3: Environment Configuration - IN PROGRESS

### 3.1 Redis Setup

Redis is required for caching and real-time data. You have several options:

#### Option A: Install Redis on Windows
1. Download Redis for Windows from: https://github.com/microsoftarchive/redis/releases
2. Install and start Redis server
3. Test connection: `redis-cli ping`

#### Option B: Use Docker (Recommended)
```bash
# Pull and run Redis container
docker run -d --name redis-sentiment -p 6379:6379 redis:alpine

# Test connection
docker exec redis-sentiment redis-cli ping
```

#### Option C: Use Redis Cloud (Free tier available)
1. Sign up at https://redis.com/
2. Create a free database
3. Update configuration with your Redis URL

### 3.2 API Keys Setup (Optional but Recommended)

For enhanced sentiment accuracy, set up the following API keys:

#### Twitter API
1. Go to https://developer.twitter.com/en/portal/dashboard
2. Create a new app
3. Get your API keys and tokens
4. Set environment variables:
   ```bash
   set TWITTER_API_KEY=your_api_key
   set TWITTER_API_SECRET=your_api_secret
   set TWITTER_BEARER_TOKEN=your_bearer_token
   ```

#### Reddit API
1. Go to https://www.reddit.com/prefs/apps
2. Create a new app
3. Get your client ID and secret
4. Set environment variables:
   ```bash
   set REDDIT_CLIENT_ID=your_client_id
   set REDDIT_CLIENT_SECRET=your_client_secret
   ```

#### News API
1. Go to https://newsapi.org/register
2. Get your API key
3. Set environment variable:
   ```bash
   set NEWS_API_KEY=your_api_key
   ```

### 3.3 Environment Variables

Create a `.env` file in the backend directory with the following content:

```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=alphapulse
DB_USER=alpha_emon
DB_PASSWORD=Emon_@17711

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# API Keys (fill in your actual keys)
TWITTER_API_KEY=your_twitter_api_key_here
TWITTER_API_SECRET=your_twitter_api_secret_here
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
NEWS_API_KEY=your_news_api_key_here

# Sentiment Model Configuration
SENTIMENT_MODEL=distilbert-base-uncased-finetuned-sst-2-english
SENTIMENT_DEVICE=auto
SENTIMENT_MAX_LENGTH=512

# Processing Configuration
SENTIMENT_COLLECTION_INTERVAL=60
SENTIMENT_AGGREGATION_INTERVAL=300
SENTIMENT_CACHE_TIMEOUT=300

# Alert Configuration
SENTIMENT_SPIKE_THRESHOLD=0.3
```

## 🧪 Step 4: Test Integration

Once Redis is running and environment variables are set, run the comprehensive test suite:

```bash
python scripts/test_enhanced_sentiment_system.py
```

This will test:
- Database connectivity
- Redis connectivity
- Sentiment analysis models
- Data collection and aggregation
- API endpoints
- Performance benchmarks

## 🚀 Step 5: Deploy to Production

After successful testing, deploy the enhanced sentiment analysis system:

1. **Start the sentiment service:**
   ```bash
   python -m app.services.enhanced_sentiment_service
   ```

2. **Start the API server:**
   ```bash
   uvicorn app.api.routes.enhanced_sentiment_api:app --host 0.0.0.0 --port 8000
   ```

3. **Integrate with frontend:**
   - Update frontend components to use new sentiment endpoints
   - Deploy the enhanced sentiment dashboard

## 📊 Current Status

- ✅ Database Migration: COMPLETED
- ✅ Dependencies Installation: COMPLETED
- 🔧 Environment Configuration: IN PROGRESS
- ⏳ Test Integration: PENDING
- ⏳ Deploy to Production: PENDING

## 🆘 Troubleshooting

### Redis Connection Issues
- Ensure Redis server is running
- Check Redis port (default: 6379)
- Verify firewall settings

### Database Connection Issues
- Ensure PostgreSQL/TimescaleDB is running
- Verify database credentials
- Check database permissions

### API Key Issues
- Verify API keys are correctly set
- Check API rate limits
- Ensure proper API permissions

### Model Loading Issues
- Check internet connection for model downloads
- Verify sufficient disk space
- Ensure PyTorch is properly installed

## 📞 Support

If you encounter any issues during setup, please:
1. Check the logs for error messages
2. Verify all dependencies are installed
3. Ensure all services are running
4. Test individual components separately
