# Environment Variables Template

Copy the content below to create your `.env` file at the project root.

```bash
# AlphaPulse Monorepo Environment Variables

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
DATABASE_URL=postgresql://alpha_emon:Emon_%4017711@localhost:55433/alphapulse
REDIS_URL=redis://localhost:6379

# =============================================================================
# BACKEND CONFIGURATION
# =============================================================================
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
DEBUG=false
LOG_LEVEL=INFO
SECRET_KEY=your_secret_key_here_change_in_production

# =============================================================================
# FRONTEND CONFIGURATION
# =============================================================================
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# =============================================================================
# API KEYS (Read-Only for Market Data)
# =============================================================================
COINGLASS_API_KEY=your_coinglass_api_key
POLYGON_API_KEY=your_polygon_api_key
COINMARKETCAP_API_KEY=your_coinmarketcap_api_key
NEWS_API_KEY=your_news_api_key
TWITTER_API_KEY=your_twitter_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key

# =============================================================================
# NOTIFICATION CONFIGURATION
# =============================================================================
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
DISCORD_WEBHOOK_URL=your_discord_webhook_url
```

