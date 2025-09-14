# 🌟 AlphaPulse Environment Configuration Setup

This guide explains how to set up your environment configuration for the AlphaPulse Candlestick Analysis System.

## 📁 Files Created

1. **`config.env.template`** - Template file with all required environment variables
2. **`backend/app/core/config.py`** - Python configuration loader with type safety
3. **`setup_env.py`** - Automated setup script to create your `.env` file
4. **`ENV_SETUP_README.md`** - This documentation file

## 🚀 Quick Setup

### Option 1: Automated Setup (Recommended)

1. **Run the setup script:**
   ```bash
   python setup_env.py
   ```

2. **Choose option 1** to create a new `.env` file
3. **Follow the prompts** to complete the setup
4. **Edit the `.env` file** with your actual API keys and credentials

### Option 2: Manual Setup

1. **Copy the template:**
   ```bash
   cp config.env.template .env
   ```

2. **Edit the `.env` file** with your actual values
3. **Generate secure secret keys** for security

## 🔑 API Keys Already Configured

The following API keys are **already configured** in the template:

- ✅ **CoinGecko API**: `CG-QQcUzHcQwbnTC7pxLAwKdwBZ`
- ✅ **Hugging Face API**: `your_huggingface_api_key_here`
- ✅ **News API**: `9d9a3e710a0a454f8bcee7e4f04e3c24`
- ✅ **Twitter API Key**: `CjHIjcq4454CKNQBvjH37XuRl`
- ✅ **Twitter API Secret**: `iiQZzQiibAPRuXA6Mh4oVnBua0Jp6dYRgbKtwiAGGVGbbXmS4h`

## 🔧 APIs to Configure Later

The following APIs will be provided later:

- 🔄 **Reddit API** - Client ID and Secret
- 🔄 **Telegram Bot API** - Bot Token
- 🔄 **Binance API** - API Key and Secret (using public webhooks for now)

## 🗄️ Database Configuration

### TimescaleDB (Open-Source Self-Hosted)

- **Host**: `localhost` (default)
- **Port**: `5432` (default)
- **Database**: `alphapulse` (default)
- **Username**: `alphapulse_user` (default)
- **Password**: **You must set this**

### Redis (For Caching - Will use later)

- **Host**: `localhost` (default)
- **Port**: `6379` (default)
- **Password**: Optional (set if required)

## 🔐 Security Configuration

The setup script automatically generates:

- **Secret Key**: 64-character random string for app security
- **JWT Secret Key**: 64-character random string for authentication

## 📊 Trading Configuration

Default values are set for:

- **Timeframes**: `["1m", "5m", "15m", "1h", "4h", "1d"]`
- **Pattern Detection**: Confidence threshold, volume confirmation, lookback period
- **Technical Indicators**: RSI, MACD, Bollinger Bands parameters
- **Risk Management**: Position sizing, risk per trade

## 🔔 Notification Services

Configure these services for trading alerts:

- **Discord**: Webhook URL, Bot Token, Guild ID, Channel ID
- **Telegram**: Bot Token, Chat ID
- **Email**: SMTP settings (Gmail recommended)

## 🚀 Usage in Your Code

### Python Backend

```python
from backend.app.core.config import get_settings

settings = get_settings()

# Access API keys
coingecko_key = settings.api.coingecko_api_key
news_api_key = settings.api.news_api_key

# Access database configuration
db_url = settings.database.timescaledb_url
redis_url = settings.database.redis_url

# Access trading parameters
timeframes = settings.trading.default_timeframes
rsi_period = settings.trading.rsi_period
```

### Frontend (Next.js)

```typescript
// Access environment variables
const coingeckoKey = process.env.NEXT_PUBLIC_COINGECKO_API_KEY;
const newsApiKey = process.env.NEXT_PUBLIC_NEWS_API_KEY;
```

## ✅ Validation

### Validate Configuration

1. **Run the validation script:**
   ```bash
   python setup_env.py
   ```
   Choose option 2 to validate your setup

2. **Or run the config module directly:**
   ```bash
   python -m backend.app.core.config
   ```

### Check for Warnings

The validation will warn you about:
- Missing API keys
- Default passwords still in place
- Unconfigured services

## 🚨 Important Security Notes

1. **Never commit `.env` files** to version control
2. **Keep API keys secure** and private
3. **Use different keys** for development and production
4. **Rotate keys regularly** for security
5. **Monitor API usage** to stay within rate limits

## 📋 Setup Checklist

- [ ] Run `python setup_env.py`
- [ ] Create `.env` file from template
- [ ] Update TimescaleDB password
- [ ] Configure Discord webhook (optional)
- [ ] Configure Telegram bot (when available)
- [ ] Set up email service (optional)
- [ ] Configure Binance API (when needed)
- [ ] Test configuration validation
- [ ] Verify all warnings are resolved

## 🆘 Troubleshooting

### Common Issues

1. **Template file not found**
   - Ensure `config.env.template` exists in the project root

2. **Permission denied**
   - Check file permissions and ensure you can write to the directory

3. **Configuration validation fails**
   - Review the warnings and update missing values

4. **API rate limits exceeded**
   - Check your API usage and implement proper caching

### Getting Help

- Check the configuration validation output
- Review the API documentation for each service
- Ensure all required environment variables are set
- Verify database connection strings are correct

## 🔄 Updates and Maintenance

- **Regular updates**: Review and update API keys monthly
- **Security audit**: Check for exposed credentials quarterly
- **Performance monitoring**: Monitor API rate limits and response times
- **Backup configuration**: Keep secure backups of your configuration

---

**Next Step**: After setting up your environment, proceed with implementing the candlestick analysis system using the files outlined in `docs/candlestick_analysis.md`.
