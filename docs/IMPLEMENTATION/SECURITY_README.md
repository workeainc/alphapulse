# AlphaPulse Security Configuration Guide

## Overview
This document outlines the security improvements implemented in AlphaPulse and provides guidance on secure configuration and deployment.

## Security Improvements Implemented

### 1. Environment Variable Management
- **Removed hardcoded credentials** from configuration files
- **Implemented proper environment variable loading** using `python-dotenv`
- **Secure defaults** for all configuration parameters
- **Environment-specific configuration** support

### 2. Input Validation and Sanitization
- **Path parameter validation** for trading symbols (alphanumeric only, max 20 chars)
- **Query parameter validation** for timeframes and limits
- **Comprehensive input sanitization** to prevent injection attacks
- **Rate limiting** and parameter bounds checking

### 3. API Security Enhancements
- **Dependency injection** instead of global variables
- **Proper error handling** with generic error messages in production
- **Input validation** for all API endpoints
- **Secure WebSocket handling** with proper connection management

### 4. Docker Security
- **Environment variable-based configuration** instead of hardcoded values
- **Secure defaults** for database credentials
- **Proper health checks** and container management

## Environment Setup

### 1. Create Environment File
Copy the template and create your `.env` file:

```bash
cp .env.example .env
```

### 2. Configure Secure Values
Update the `.env` file with your actual values:

```bash
# Database Configuration
TIMESCALEDB_HOST=localhost
TIMESCALEDB_PORT=5432
TIMESCALEDB_DATABASE=alphapulse
TIMESCALEDB_USERNAME=your_secure_username
TIMESCALEDB_PASSWORD=your_secure_password
DATABASE_URL=postgresql://your_secure_username:your_secure_password@localhost:5432/alphapulse

# Application Security
DEBUG=false
LOG_LEVEL=INFO
SECRET_KEY=your_very_long_and_random_secret_key_here
JWT_SECRET_KEY=your_very_long_and_random_jwt_secret_key_here

# API Keys (set your actual keys)
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
```

### 3. Generate Secure Keys
Use a secure method to generate secret keys:

```bash
# Generate a secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate a secure JWT secret
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Security Best Practices

### 1. Production Deployment
- **Never commit `.env` files** to version control
- **Use environment variables** or secure secret management in production
- **Set `DEBUG=false`** in production environments
- **Use HTTPS** for all external communications
- **Implement proper firewall rules** and network segmentation

### 2. Database Security
- **Use strong passwords** for database accounts
- **Limit database access** to necessary IP addresses only
- **Regular security updates** for database software
- **Encrypt sensitive data** at rest and in transit

### 3. API Security
- **Implement rate limiting** to prevent abuse
- **Use API keys** for external integrations
- **Validate all inputs** and sanitize data
- **Log security events** and monitor for suspicious activity

### 4. Monitoring and Logging
- **Monitor system resources** and performance
- **Log security events** and access attempts
- **Set up alerts** for unusual activity
- **Regular security audits** and penetration testing

## Configuration Validation

### 1. Test Environment Variables
Verify that environment variables are loaded correctly:

```bash
python -c "from app.core.unified_config import get_settings; print(get_settings().DATABASE_URL)"
```

### 2. Test API Endpoints
Test input validation with various inputs:

```bash
# Test valid symbol
curl "http://localhost:8000/api/candlestick/patterns/BTC"

# Test invalid symbol (should fail)
curl "http://localhost:8000/api/candlestick/patterns/INVALID_SYMBOL!"

# Test invalid timeframe (should fail)
curl "http://localhost:8000/api/candlestick/patterns/BTC?timeframe=invalid"
```

### 3. Test Docker Configuration
Verify Docker containers use environment variables:

```bash
docker-compose up --build
docker-compose logs dashboard
```

## Troubleshooting

### 1. Environment Variable Issues
- Ensure `.env` file exists and is readable
- Check that `python-dotenv` is installed
- Verify environment variable names match configuration

### 2. Database Connection Issues
- Verify database credentials in `.env` file
- Check database service is running
- Ensure network connectivity between services

### 3. API Validation Issues
- Check input format matches validation rules
- Verify error messages are appropriate
- Test with various input combinations

## Security Checklist

- [ ] Environment variables configured securely
- [ ] Hardcoded credentials removed
- [ ] Input validation implemented
- [ ] Error handling secure (no information leakage)
- [ ] Docker configuration secure
- [ ] Production environment variables set
- [ ] Security monitoring enabled
- [ ] Regular security updates scheduled

## Additional Resources

- [FastAPI Security Documentation](https://fastapi.tiangolo.com/tutorial/security/)
- [OWASP Security Guidelines](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python-security.readthedocs.io/)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)

## Support

For security-related issues or questions:
1. Review this documentation
2. Check the application logs
3. Verify environment configuration
4. Contact the development team

**Remember: Security is an ongoing process. Regularly review and update your security configuration.**
