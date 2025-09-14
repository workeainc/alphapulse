# 🛡️ Risk Management & Bot Protection

## Overview
Keeps AlphaPulse safe from market chaos and technical failures. This layer implements comprehensive risk controls, position limits, and self-diagnostics to protect capital and ensure system stability.

## ✅ Implemented Components

### 1. Risk Manager
- **File**: `backend/app/services/risk_manager.py` ✅
- **File**: `backend/execution/risk_manager.py` ✅
- **Features**:
  - Position sizing calculation
  - Risk validation
  - Portfolio exposure management
  - Basic daily loss limits

### 2. Database Models
- **File**: `backend/database/models.py` ✅
- **Features**:
  - Trade history tracking
  - Portfolio performance
  - Risk metrics storage

## 🚧 Partially Implemented

### 3. Position Limits
- **Status**: Basic structure exists
- **Needs**: Advanced position management

### 4. Portfolio Exposure
- **Status**: Basic calculation exists
- **Needs**: Real-time monitoring

## ❌ Not Yet Implemented

### 5. Advanced Risk Controls
- **Required**: Max daily loss limits
- **Purpose**: Prevent catastrophic losses
- **Priority**: High

### 6. Consecutive Loss Protection
- **Required**: Pause trading after N losses
- **Purpose**: Prevent tilt trading
- **Priority**: High

### 7. News Event Filter
- **Required**: Skip trades before big events
- **Purpose**: Avoid volatility spikes
- **Priority**: Medium

### 8. Self-Diagnostics
- **Required**: Check API latency, data errors
- **Purpose**: Auto-pause on faults
- **Priority**: High

### 9. Alert System
- **Required**: Telegram/Discord notifications
- **Purpose**: Real-time monitoring
- **Priority**: Medium

## 🔧 Implementation Tasks

### Immediate (This Week)
1. **Enhanced Risk Manager**
   ```python
   # Update: backend/app/services/risk_manager.py
   class RiskManager:
       def __init__(self):
           self.max_daily_loss = -0.05  # 5% max daily loss
           self.max_consecutive_losses = 3
           self.max_open_positions = 5
           self.max_portfolio_exposure = 0.20  # 20% max exposure
           self.consecutive_losses = 0
           self.daily_pnl = 0.0
       
       def check_daily_loss_limit(self) -> bool:
           """Check if daily loss limit exceeded"""
           return self.daily_pnl >= self.max_daily_loss
       
       def check_consecutive_losses(self) -> bool:
           """Check if consecutive loss limit exceeded"""
           return self.consecutive_losses < self.max_consecutive_losses
       
       def check_position_limit(self, current_positions: int) -> bool:
           """Check if position limit exceeded"""
           return current_positions < self.max_open_positions
       
       def check_portfolio_exposure(self, current_exposure: float) -> bool:
           """Check if portfolio exposure exceeded"""
           return current_exposure <= self.max_portfolio_exposure
   ```

2. **News Event Filter**
   ```python
   # New file: backend/protection/news_filter.py
   class NewsEventFilter:
       def __init__(self):
           self.important_events = [
               "FOMC", "CPI", "NFP", "GDP", "Fed Meeting",
               "ECB Meeting", "BOE Meeting", "BOJ Meeting"
           ]
           self.event_buffer_hours = 2  # Hours before/after event
       
       def is_high_impact_event_soon(self, current_time: datetime) -> bool:
           """Check if high-impact event is coming soon"""
           # Implementation needed: Calendar API integration
           pass
       
       def should_skip_trading(self, symbol: str) -> bool:
           """Determine if trading should be skipped"""
           # Implementation needed: Event impact assessment
           pass
   ```

### Short Term (Next 2 Weeks)
1. **Self-Diagnostics System**
   - API health monitoring
   - Data quality checks
   - System performance metrics

2. **Alert System**
   - Telegram bot integration
   - Discord webhook setup
   - Email notifications

### Medium Term (Next Month)
1. **Advanced Risk Models**
   - VaR calculations
   - Stress testing
   - Correlation analysis

## 📊 Risk Management Architecture

### Risk Check Flow
```
Signal → Risk Validation → Position Check → Exposure Check → Execution
  ↓           ↓              ↓              ↓            ↓
Strategy   Daily Loss    Position      Portfolio    Order
Manager    Limit        Limit         Exposure     Place
```

### Risk Hierarchy
```
Portfolio Level → Symbol Level → Trade Level
      ↓              ↓            ↓
   Exposure      Position      Stop Loss
   Limits        Limits        Limits
```

## 🎯 Risk Controls

### Daily Loss Limits
```python
class DailyLossManager:
    def __init__(self):
        self.max_daily_loss_pct = -0.05  # 5% max daily loss
        self.daily_pnl = 0.0
        self.daily_start_balance = 0.0
        self.trading_enabled = True
    
    def update_daily_pnl(self, trade_pnl: float):
        """Update daily PnL"""
        self.daily_pnl += trade_pnl
        
        # Check if daily limit exceeded
        if self.daily_pnl <= self.max_daily_loss_pct:
            self.trading_enabled = False
            logger.warning("Daily loss limit exceeded - trading disabled")
    
    def reset_daily_tracking(self):
        """Reset daily tracking at market open"""
        self.daily_pnl = 0.0
        self.daily_start_balance = self.get_current_balance()
        self.trading_enabled = True
```

### Consecutive Loss Protection
```python
class ConsecutiveLossManager:
    def __init__(self):
        self.max_consecutive_losses = 3
        self.consecutive_losses = 0
        self.trading_enabled = True
    
    def record_trade_result(self, trade_pnl: float):
        """Record trade result and update consecutive losses"""
        if trade_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Check if limit exceeded
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.trading_enabled = False
            logger.warning("Consecutive loss limit exceeded - trading paused")
    
    def should_allow_trading(self) -> bool:
        """Check if trading should be allowed"""
        return self.trading_enabled
```

### Position Limits
```python
class PositionLimitManager:
    def __init__(self):
        self.max_open_positions = 5
        self.max_position_size_pct = 0.10  # 10% max per position
        self.max_correlation_exposure = 0.30  # 30% max correlated exposure
    
    def check_position_limit(self, current_positions: int) -> bool:
        """Check if position limit exceeded"""
        return current_positions < self.max_open_positions
    
    def check_position_size(self, position_value: float, portfolio_value: float) -> bool:
        """Check if position size is within limits"""
        position_pct = position_value / portfolio_value
        return position_pct <= self.max_position_size_pct
    
    def check_correlation_exposure(self, new_symbol: str, existing_positions: List[Position]) -> bool:
        """Check if new position would exceed correlation limits"""
        # Implementation needed: Correlation calculation
        pass
```

## 🔒 Portfolio Exposure Management

### Exposure Calculation
```python
class ExposureManager:
    def __init__(self):
        self.max_total_exposure = 0.20  # 20% max total exposure
        self.max_sector_exposure = 0.15  # 15% max per sector
        self.max_currency_exposure = 0.25  # 25% max per currency
    
    def calculate_total_exposure(self, positions: List[Position]) -> float:
        """Calculate total portfolio exposure"""
        total_exposure = 0.0
        for position in positions:
            total_exposure += abs(position.notional_value)
        
        return total_exposure
    
    def calculate_sector_exposure(self, positions: List[Position]) -> Dict[str, float]:
        """Calculate exposure per sector"""
        sector_exposure = {}
        for position in positions:
            sector = self.get_symbol_sector(position.symbol)
            if sector not in sector_exposure:
                sector_exposure[sector] = 0.0
            sector_exposure[sector] += abs(position.notional_value)
        
        return sector_exposure
    
    def check_exposure_limits(self, new_position: Position, 
                             existing_positions: List[Position]) -> bool:
        """Check if new position would exceed exposure limits"""
        # Check total exposure
        total_exposure = self.calculate_total_exposure(existing_positions + [new_position])
        if total_exposure > self.max_total_exposure:
            return False
        
        # Check sector exposure
        sector_exposure = self.calculate_sector_exposure(existing_positions + [new_position])
        for sector, exposure in sector_exposure.items():
            if exposure > self.max_sector_exposure:
                return False
        
        return True
```

## 🚨 Self-Diagnostics System

### System Health Monitoring
```python
class SystemHealthMonitor:
    def __init__(self):
        self.health_checks = {}
        self.critical_errors = []
        self.trading_enabled = True
    
    async def run_health_checks(self):
        """Run all health checks"""
        checks = [
            self.check_database_connection(),
            self.check_api_latency(),
            self.check_data_quality(),
            self.check_system_resources()
        ]
        
        results = await asyncio.gather(*checks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.critical_errors.append(str(result))
                self.trading_enabled = False
        
        return self.trading_enabled
    
    async def check_database_connection(self) -> bool:
        """Check database connection health"""
        try:
            # Implementation needed: Database ping
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    async def check_api_latency(self) -> bool:
        """Check API latency"""
        try:
            # Implementation needed: API response time check
            return True
        except Exception as e:
            logger.error(f"API latency check failed: {e}")
            return False
    
    async def check_data_quality(self) -> bool:
        """Check data quality"""
        try:
            # Implementation needed: Data validation
            return True
        except Exception as e:
            logger.error(f"Data quality check failed: {e}")
            return False
    
    async def check_system_resources(self) -> bool:
        """Check system resources"""
        try:
            # Implementation needed: CPU, memory, disk check
            return True
        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            return False
```

## 📱 Alert System

### Telegram Bot Integration
```python
# New file: backend/protection/alerts.py
class AlertManager:
    def __init__(self):
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        self.alerts_enabled = True
    
    async def send_telegram_alert(self, message: str):
        """Send alert via Telegram"""
        if not self.alerts_enabled or not self.telegram_bot_token:
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status != 200:
                        logger.error(f"Telegram alert failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
    
    async def send_discord_alert(self, message: str):
        """Send alert via Discord"""
        if not self.alerts_enabled or not self.discord_webhook_url:
            return
        
        try:
            data = {"content": message}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.discord_webhook_url, json=data) as response:
                    if response.status != 204:
                        logger.error(f"Discord alert failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error sending Discord alert: {e}")
    
    async def send_critical_alert(self, message: str):
        """Send critical alert via all channels"""
        critical_message = f"🚨 CRITICAL: {message}"
        
        await asyncio.gather(
            self.send_telegram_alert(critical_message),
            self.send_discord_alert(critical_message),
            return_exceptions=True
        )
```

## 🔄 Risk Management Flow

### Pre-Trade Risk Check
```python
async def pre_trade_risk_check(self, signal: Signal) -> RiskCheckResult:
    """Perform pre-trade risk checks"""
    try:
        # 1. Check daily loss limit
        if not self.risk_manager.check_daily_loss_limit():
            return RiskCheckResult(
                allowed=False,
                reason="Daily loss limit exceeded"
            )
        
        # 2. Check consecutive losses
        if not self.risk_manager.check_consecutive_losses():
            return RiskCheckResult(
                allowed=False,
                reason="Consecutive loss limit exceeded"
            )
        
        # 3. Check position limits
        current_positions = len(self.get_open_positions())
        if not self.risk_manager.check_position_limit(current_positions):
            return RiskCheckResult(
                allowed=False,
                reason="Position limit exceeded"
            )
        
        # 4. Check news events
        if self.news_filter.should_skip_trading(signal.symbol):
            return RiskCheckResult(
                allowed=False,
                reason="High-impact news event approaching"
            )
        
        # 5. Check system health
        if not await self.health_monitor.run_health_checks():
            return RiskCheckResult(
                allowed=False,
                reason="System health check failed"
            )
        
        return RiskCheckResult(allowed=True, reason="All checks passed")
        
    except Exception as e:
        logger.error(f"Risk check failed: {e}")
        return RiskCheckResult(
            allowed=False,
            reason=f"Risk check error: {str(e)}"
        )
```

## 📊 Risk Metrics

### Risk Metrics
- **Daily PnL**: Current day's profit/loss
- **Consecutive Losses**: Number of consecutive losing trades
- **Portfolio Exposure**: Total market exposure percentage
- **Position Concentration**: Largest single position size
- **Correlation Risk**: Exposure to correlated assets

### Performance Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest portfolio decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss

## 🚀 Next Steps

1. **Implement daily loss limits** with automatic trading pause
2. **Add consecutive loss protection** to prevent tilt trading
3. **Create news event filter** for high-impact events
4. **Set up self-diagnostics** for system health monitoring
5. **Integrate alert system** (Telegram/Discord)

## 📚 Related Documentation

- [Data Collection Layer](./01_data_collection_layer.md)
- [Storage & Processing Layer](./02_storage_processing_layer.md)
- [Analysis Layer](./03_analysis_layer.md)
- [Execution Layer](./04_execution_layer.md)
- [Pine Script Integration](./06_pine_script_integration.md)
