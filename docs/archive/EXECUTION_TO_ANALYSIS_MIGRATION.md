# AlphaPulse Migration Guide: Execution to Analysis Engine

## Overview

This document outlines the major refactoring of AlphaPulse from a trading execution platform to a pure signal analysis and recommendation engine.

**Migration Date:** October 26, 2025  
**Version:** 1.0.0 → 2.0.0  
**Status:** Complete

---

## What Changed and Why

### Core Concept Clarification

**AlphaPulse is a Signal Analysis & Recommendation Engine, NOT a Trading Execution Platform.**

The system was originally designed with trade execution capabilities, but the core concept has been clarified:

**What AlphaPulse DOES:**
- Analyzes real-time market data
- Generates high-confidence trading signals
- Provides risk recommendations (SL/TP/position sizing)
- Delivers alerts and notifications
- Tracks signal performance for ML validation

**What AlphaPulse DOES NOT DO:**
- Execute trades on exchanges
- Manage real positions
- Place orders automatically
- Handle exchange credentials for trading
- Track actual P&L from executed trades

---

## Architectural Changes

### 1. Archived Execution Components

The following components have been moved to `backend/archived/`:

#### Execution Module (`backend/archived/execution/`)
- `order_manager.py` - Order placement logic
- `exchange_trading_connector.py` - Exchange API integration for trading
- `portfolio_manager.py` - Position management
- `position_scaling_manager.py` - Position scaling logic
- `sl_tp_manager.py` - Stop-loss/take-profit management
- `advanced_portfolio_manager.py` - Advanced portfolio tracking
- `advanced_risk_manager.py` - Execution-level risk management
- `funding_rate_executor.py` - Funding rate trade execution
- `execution_analytics.py` - Execution performance tracking

#### Services (`backend/archived/services/`)
- `trading_engine.py` - Old trading execution engine
- `core_trading_engine.py` - Core trading orchestrator
- `enhanced_trading_engine.py` - Enhanced trading engine

#### Config (`backend/archived/config/`)
- `exchange_config.py` - Exchange credentials management

### 2. Renamed Components

| Old Name | New Name | Purpose |
|----------|----------|---------|
| `Trade` (model) | `SignalRecommendation` | Stores recommendations, not executed trades |
| `TradingEngine` | `SignalOrchestrator` | Orchestrates signal generation, not trade execution |
| `PaperTradingEngine` | `SignalOutcomeTracker` | Tracks hypothetical outcomes for ML validation |
| `trades` (table) | `signal_recommendations` (table) | Database table for recommendations |

### 3. Database Schema Changes

#### Signal Recommendations Table

**Old table:** `trades`  
**New table:** `signal_recommendations`

#### Column Renames:
- `entry_price` → `suggested_entry_price`
- `exit_price` → `suggested_exit_price`
- `quantity` → `suggested_quantity`
- `leverage` → `suggested_leverage`
- `pnl` → `hypothetical_pnl`
- `pnl_percentage` → `hypothetical_pnl_percentage`
- `stop_loss` → `suggested_stop_loss`
- `take_profit` → `suggested_take_profit`
- `trailing_stop` → `suggested_trailing_stop`
- `entry_time` → `recommendation_time`
- `exit_time` → `expiry_time`

#### Status Values:
- `'open'` → `'pending'`
- `'closed'` → `'user_executed'`
- `'cancelled'` → `'cancelled'` (unchanged)
- New: `'expired'`

---

## Migration Guide

### For Developers

#### 1. Update Imports

**Old:**
```python
from database.models import Trade
from execution.order_manager import OrderManager
from app.services.trading_engine import TradingEngine
```

**New:**
```python
from database.models import SignalRecommendation
# execution imports removed - use signal_orchestrator instead
from app.services.signal_orchestrator import SignalOrchestrator
```

#### 2. Update Model Usage

**Old:**
```python
trade = Trade(
    symbol="BTCUSDT",
    side="long",
    entry_price=45000,
    quantity=0.1,
    stop_loss=44000,
    take_profit=47000
)
```

**New:**
```python
recommendation = SignalRecommendation(
    symbol="BTCUSDT",
    side="long",
    suggested_entry_price=45000,
    suggested_quantity=0.1,
    suggested_stop_loss=44000,
    suggested_take_profit=47000,
    status='pending'
)
```

#### 3. Run Database Migration

```bash
cd backend/database/migrations
python rename_trades_to_recommendations.py
```

To rollback:
```bash
python rename_trades_to_recommendations.py down
```

#### 4. Update API Endpoints

**Old endpoints:**
- `GET /api/trades`
- `POST /api/trades/execute`

**New endpoints:**
- `GET /api/recommendations`
- `POST /api/signals/generate`

#### 5. Update Frontend Components

**Old:**
```typescript
interface Trade {
  entry_price: number;
  stop_loss: number;
  status: 'open' | 'closed';
}
```

**New:**
```typescript
interface SignalRecommendation {
  suggested_entry_price: number;
  suggested_stop_loss: number;
  status: 'pending' | 'user_executed' | 'expired' | 'cancelled';
}
```

### For Users

#### 1. Signal Workflow

**Old workflow:**
1. Signal generated
2. Trade executed automatically
3. Position managed by system

**New workflow:**
1. Signal generated with recommendations
2. User reviews signal and parameters
3. User executes trade manually on their exchange
4. User can report outcome (optional) for ML validation

#### 2. API Changes

Signal recommendations now include all parameters needed for manual execution:

```json
{
  "signal_id": "sig_BTCUSDT_1635123456",
  "symbol": "BTCUSDT",
  "side": "long",
  "suggested_entry_price": 45000,
  "suggested_stop_loss": 44000,
  "suggested_take_profit": 47000,
  "suggested_quantity": 0.1,
  "suggested_leverage": 1,
  "confidence": 0.85,
  "strategy_name": "ml_ensemble",
  "market_regime": "trending_bullish",
  "recommendation_time": "2025-10-26T12:00:00Z",
  "status": "pending"
}
```

#### 3. No Exchange Credentials Needed

AlphaPulse no longer requires exchange API keys for trading. You only need:
- Market data API keys (read-only)
- News/sentiment API keys (read-only)

---

## Backward Compatibility

### Deprecation Notices

The following have backward compatibility aliases with deprecation warnings:

```python
# Will work but show deprecation warning
from database.models import Trade  # → SignalRecommendation
from trading.paper_trading_engine import PaperTradingEngine  # → SignalOutcomeTracker
```

**These aliases will be removed in version 3.0.0 (Q1 2026)**

### Execution Module

The `backend/execution/` module now shows a deprecation warning when imported:

```python
import execution  # DeprecationWarning
```

All execution functionality has been archived.

---

## Testing After Migration

### 1. Verify Database Migration

```python
# Test that signal_recommendations table exists
from database.models import SignalRecommendation
from database.connection import get_db

async with get_db() as db:
    count = await db.query(SignalRecommendation).count()
    print(f"Signal recommendations in database: {count}")
```

### 2. Test Signal Generation

```python
from app.services.signal_orchestrator import SignalOrchestrator

orchestrator = SignalOrchestrator()
await orchestrator.start()

# Check that signals are being generated
metrics = await orchestrator.get_performance_metrics()
print(f"Signals generated: {metrics['signals_generated']}")
```

### 3. Verify No Execution Attempts

```python
# This should NOT exist anymore
try:
    from execution.order_manager import OrderManager
    print("ERROR: Execution components still active!")
except (ImportError, DeprecationWarning):
    print("✓ Execution components properly archived")
```

---

## Benefits of Migration

1. **Clearer Purpose:** System focus is on analysis, not execution
2. **Reduced Complexity:** ~4000 lines of execution code removed
3. **Lower Risk:** No financial risk from automated execution
4. **Better Focus:** Development resources on signal quality
5. **User Control:** Users maintain full control over trade execution
6. **Regulatory Compliance:** Avoids automated trading regulations

---

## Support and Questions

### Common Questions

**Q: Can I still use AlphaPulse for automated trading?**  
A: AlphaPulse provides signals and recommendations. You can build your own execution layer or use the archived components as reference.

**Q: What happened to my trade history?**  
A: Historical data has been migrated from `trades` to `signal_recommendations` table. All data is preserved.

**Q: Can I access the old execution code?**  
A: Yes, all execution components are archived in `backend/archived/` for reference.

**Q: Will paper trading still work?**  
A: Yes, renamed to `SignalOutcomeTracker` - it tracks hypothetical outcomes for ML validation.

### Getting Help

- **Documentation:** Check `/docs` folder for detailed guides
- **Issues:** Report bugs via GitHub Issues
- **Discord:** Join our community for support

---

## Rollback Procedure

If you need to rollback (not recommended):

```bash
# 1. Rollback database
cd backend/database/migrations
python rename_trades_to_recommendations.py down

# 2. Restore archived files
# Manually copy files from backend/archived/ back to original locations

# 3. Revert git commit
git revert <migration-commit-hash>
```

---

**Migration completed:** October 26, 2025  
**Document version:** 1.0  
**Last updated:** October 26, 2025

