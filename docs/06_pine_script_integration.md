# 📊 Pine Script Integration

## Overview
Custom TradingView Pine Script integration for real-time signal generation and automated trading. This layer connects AlphaPulse to TradingView's powerful charting platform for advanced technical analysis and signal execution.

## ✅ Implemented Components

### 1. Pine Script Files
- **File**: `pine_scripts/alpha_pulse_indicator.pine` ✅
- **File**: `pine_scripts/breakout_detector.pine` ✅
- **File**: `pine_scripts/ema_trend.pine` ✅
- **File**: `pine_scripts/rsi_reversion.pine` ✅
- **Features**:
  - Custom indicators for AlphaPulse
  - Breakout detection algorithms
  - EMA trend analysis
  - RSI mean reversion signals

### 2. Candlestick Analysis Integration
- **File**: `backend/routes/candlestick_analysis.py` ✅
- **Features**:
  - Webhook endpoint for TradingView
  - Signal processing and validation
  - Integration with analysis layer

## 🚧 Partially Implemented

### 3. Webhook Integration
- **Status**: Basic endpoint exists
- **Needs**: Complete signal processing pipeline

### 4. Signal Validation
- **Status**: Basic structure exists
- **Needs**: Advanced validation logic

## ❌ Not Yet Implemented

### 5. Real-Time Signal Processing
- **Required**: Process incoming webhook signals
- **Purpose**: Execute trades automatically
- **Priority**: High

### 6. Signal Quality Filtering
- **Required**: Filter low-quality signals
- **Purpose**: Improve trade success rate
- **Priority**: High

### 7. Multi-Timeframe Confluence
- **Required**: Validate signals across timeframes
- **Purpose**: Higher probability trades
- **Priority**: Medium

### 8. Signal History Tracking
- **Required**: Track signal performance
- **Purpose**: Strategy optimization
- **Priority**: Medium

## 🔧 Implementation Tasks

### Immediate (This Week)
1. **Enhanced Webhook Handler**
   ```python
   # Update: backend/routes/candlestick_analysis.py
   @app.post("/webhook/tradingview")
   async def tradingview_webhook(request: Request):
       """Handle TradingView webhook signals"""
       try:
           # Parse webhook data
           data = await request.json()
           
           # Validate signal
           signal = validate_tradingview_signal(data)
           if not signal:
               return {"status": "error", "message": "Invalid signal"}
           
           # Process signal
           result = await process_tradingview_signal(signal)
           
           return {"status": "success", "result": result}
           
       except Exception as e:
           logger.error(f"Webhook error: {e}")
           return {"status": "error", "message": str(e)}
   ```

2. **Signal Processor**
   ```python
   # New file: backend/services/signal_processor.py
   class TradingViewSignalProcessor:
       def __init__(self):
           self.signal_history = []
           self.quality_threshold = 0.7
           self.min_volume_confirmation = 1.5
       
       async def process_signal(self, signal: TradingViewSignal) -> SignalResult:
           """Process incoming TradingView signal"""
           try:
               # 1. Validate signal quality
               quality_score = self._calculate_signal_quality(signal)
               if quality_score < self.quality_threshold:
                   return SignalResult(
                       valid=False,
                       reason=f"Signal quality too low: {quality_score:.2f}"
                   )
               
               # 2. Check volume confirmation
               if not self._check_volume_confirmation(signal):
                   return SignalResult(
                       valid=False,
                       reason="Insufficient volume confirmation"
                   )
               
               # 3. Validate multi-timeframe confluence
               if not await self._check_timeframe_confluence(signal):
                   return SignalResult(
                       valid=False,
                       reason="Multi-timeframe confluence failed"
                   )
               
               # 4. Convert to internal signal
               internal_signal = self._convert_to_internal_signal(signal)
               
               # 5. Store signal history
               self._store_signal_history(signal, quality_score)
               
               return SignalResult(
                   valid=True,
                   signal=internal_signal,
                   quality_score=quality_score
               )
               
           except Exception as e:
               logger.error(f"Signal processing error: {e}")
               return SignalResult(
                   valid=False,
                   reason=f"Processing error: {str(e)}"
               )
   ```

### Short Term (Next 2 Weeks)
1. **Signal Quality Metrics**
   - Volume confirmation
   - Price action validation
   - Indicator confluence

2. **Multi-Timeframe Analysis**
   - Higher timeframe trend validation
   - Lower timeframe entry timing
   - Confluence scoring

### Medium Term (Next Month)
1. **Advanced Signal Filtering**
   - Machine learning quality scoring
   - Historical performance analysis
   - Dynamic threshold adjustment

## 📊 Pine Script Architecture

### Signal Flow
```
TradingView → Webhook → Signal Processor → Validation → Execution
     ↓           ↓           ↓              ↓           ↓
  Pine Script  HTTP      Quality      Multi-TF    Order
  Indicator    POST      Check        Confluence  Place
```

### Signal Components
```
Signal = {
    symbol: str,
    timeframe: str,
    signal_type: str,  # buy/sell
    price: float,
    volume: float,
    indicators: Dict,
    timestamp: datetime,
    confidence: float
}
```

## 🎯 Pine Script Indicators

### AlphaPulse Indicator
```pinescript
// File: pine_scripts/alpha_pulse_indicator.pine
//@version=5
indicator("AlphaPulse Indicator", overlay=true)

// Input parameters
rsi_length = input.int(14, "RSI Length")
ema_fast = input.int(9, "Fast EMA")
ema_slow = input.int(21, "Slow EMA")
atr_length = input.int(14, "ATR Length")

// Calculate indicators
rsi = ta.rsi(close, rsi_length)
ema_fast_val = ta.ema(close, ema_fast)
ema_slow_val = ta.ema(close, ema_slow)
atr = ta.atr(atr_length)

// Generate signals
bullish_signal = ta.crossover(ema_fast_val, ema_slow_val) and rsi < 70
bearish_signal = ta.crossunder(ema_fast_val, ema_slow_val) and rsi > 30

// Plot signals
plotshape(bullish_signal, "Bullish Signal", shape.triangleup, location.belowbar, color.green, size=size.small)
plotshape(bearish_signal, "Bearish Signal", shape.triangledown, location.abovebar, color.red, size=size.small)

// Alert conditions
alertcondition(bullish_signal, "AlphaPulse Buy Signal", "Bullish signal detected")
alertcondition(bearish_signal, "AlphaPulse Sell Signal", "Bearish signal detected")
```

### Breakout Detector
```pinescript
// File: pine_scripts/breakout_detector.pine
//@version=5
indicator("Breakout Detector", overlay=true)

// Input parameters
lookback = input.int(20, "Lookback Period")
volume_multiplier = input.float(1.5, "Volume Multiplier")

// Calculate breakout levels
highest_high = ta.highest(high, lookback)
lowest_low = ta.lowest(low, lookback)

// Detect breakouts
bullish_breakout = close > highest_high[1] and volume > volume_multiplier * ta.sma(volume, 20)
bearish_breakout = close < lowest_low[1] and volume > volume_multiplier * ta.sma(volume, 20)

// Plot breakout levels
plot(highest_high, "Resistance", color.red, linewidth=2)
plot(lowest_low, "Support", color.green, linewidth=2)

// Plot breakout signals
plotshape(bullish_breakout, "Bullish Breakout", shape.triangleup, location.belowbar, color.green, size=size.normal)
plotshape(bearish_breakout, "Bearish Breakout", shape.triangledown, location.abovebar, color.red, size=size.normal)

// Alert conditions
alertcondition(bullish_breakout, "Bullish Breakout", "Price broke above resistance with volume")
alertcondition(bearish_breakout, "Bearish Breakout", "Price broke below support with volume")
```

### EMA Trend Strategy
```pinescript
// File: pine_scripts/ema_trend.pine
//@version=5
indicator("EMA Trend Strategy", overlay=true)

// Input parameters
ema_9 = input.int(9, "EMA 9")
ema_21 = input.int(21, "EMA 21")
ema_50 = input.int(50, "EMA 50")
ema_200 = input.int(200, "EMA 200")

// Calculate EMAs
ema_9_val = ta.ema(close, ema_9)
ema_21_val = ta.ema(close, ema_21)
ema_50_val = ta.ema(close, ema_50)
ema_200_val = ta.ema(close, ema_200)

// Trend conditions
bullish_trend = ema_9_val > ema_21_val and ema_21_val > ema_50_val and ema_50_val > ema_200_val
bearish_trend = ema_9_val < ema_21_val and ema_21_val < ema_50_val and ema_50_val < ema_200_val

// Entry signals
bullish_entry = ta.crossover(ema_9_val, ema_21_val) and bullish_trend
bearish_entry = ta.crossunder(ema_9_val, ema_21_val) and bearish_trend

// Plot EMAs
plot(ema_9_val, "EMA 9", color.blue, linewidth=1)
plot(ema_21_val, "EMA 21", color.orange, linewidth=1)
plot(ema_50_val, "EMA 50", color.yellow, linewidth=2)
plot(ema_200_val, "EMA 200", color.white, linewidth=3)

// Plot signals
plotshape(bullish_entry, "Bullish Entry", shape.triangleup, location.belowbar, color.green, size=size.small)
plotshape(bearish_entry, "Bearish Entry", shape.triangledown, location.abovebar, color.red, size=size.small)

// Alert conditions
alertcondition(bullish_entry, "EMA Trend Buy", "Bullish EMA crossover in uptrend")
alertcondition(bearish_entry, "EMA Trend Sell", "Bearish EMA crossover in downtrend")
```

### RSI Reversion Strategy
```pinescript
// File: pine_scripts/rsi_reversion.pine
//@version=5
indicator("RSI Reversion Strategy", overlay=true)

// Input parameters
rsi_length = input.int(14, "RSI Length")
oversold_level = input.int(30, "Oversold Level")
overbought_level = input.int(70, "Overbought Level")
confirmation_length = input.int(3, "Confirmation Candles")

// Calculate RSI
rsi = ta.rsi(close, rsi_length)

// Reversion signals
oversold_signal = rsi < oversold_level
overbought_signal = rsi > overbought_level

// Confirmation (price action)
bullish_confirmation = oversold_signal and close > open and close > close[1]
bearish_confirmation = overbought_signal and close < open and close < close[1]

// Plot RSI levels
hline(oversold_level, "Oversold", color.green, linestyle=hline.style_dashed)
hline(overbought_level, "Overbought", color.red, linestyle=hline.style_dashed)

// Plot signals
plotshape(bullish_confirmation, "Bullish Reversion", shape.triangleup, location.belowbar, color.green, size=size.small)
plotshape(bearish_confirmation, "Bearish Reversion", shape.triangledown, location.abovebar, color.red, size=size.small)

// Alert conditions
alertcondition(bullish_confirmation, "RSI Reversion Buy", "RSI oversold with bullish confirmation")
alertcondition(bearish_confirmation, "RSI Reversion Sell", "RSI overbought with bearish confirmation")
```

## 🔄 Signal Processing Pipeline

### Webhook to Signal
```python
async def handle_tradingview_webhook(self, webhook_data: dict) -> TradingViewSignal:
    """Convert webhook data to internal signal"""
    try:
        # Extract signal data
        signal = TradingViewSignal(
            symbol=webhook_data.get("symbol"),
            timeframe=webhook_data.get("timeframe"),
            signal_type=webhook_data.get("signal_type"),
            price=float(webhook_data.get("price", 0)),
            volume=float(webhook_data.get("volume", 0)),
            indicators=webhook_data.get("indicators", {}),
            timestamp=datetime.fromisoformat(webhook_data.get("timestamp")),
            confidence=float(webhook_data.get("confidence", 0.5))
        )
        
        # Validate required fields
        if not all([signal.symbol, signal.signal_type, signal.price]):
            raise ValueError("Missing required signal fields")
        
        return signal
        
    except Exception as e:
        logger.error(f"Error parsing webhook data: {e}")
        raise
```

### Signal Quality Assessment
```python
def _calculate_signal_quality(self, signal: TradingViewSignal) -> float:
    """Calculate signal quality score (0-1)"""
    quality_score = 0.0
    
    # 1. Volume confirmation (30% weight)
    volume_score = min(signal.volume / self.avg_volume, 2.0) / 2.0
    quality_score += volume_score * 0.3
    
    # 2. Price action confirmation (25% weight)
    price_action_score = self._assess_price_action(signal)
    quality_score += price_action_score * 0.25
    
    # 3. Indicator confluence (25% weight)
    indicator_score = self._assess_indicator_confluence(signal)
    quality_score += indicator_score * 0.25
    
    # 4. Market regime alignment (20% weight)
    regime_score = self._assess_market_regime_alignment(signal)
    quality_score += regime_score * 0.20
    
    return min(quality_score, 1.0)

def _assess_price_action(self, signal: TradingViewSignal) -> float:
    """Assess price action confirmation"""
    # Implementation needed: Candlestick pattern analysis
    return 0.8  # Placeholder

def _assess_indicator_confluence(self, signal: TradingViewSignal) -> float:
    """Assess indicator confluence"""
    # Implementation needed: Multiple indicator agreement
    return 0.7  # Placeholder

def _assess_market_regime_alignment(self, signal: TradingViewSignal) -> float:
    """Assess alignment with current market regime"""
    # Implementation needed: Market regime detection
    return 0.9  # Placeholder
```

### Multi-Timeframe Validation
```python
async def _check_timeframe_confluence(self, signal: TradingViewSignal) -> bool:
    """Check multi-timeframe confluence"""
    try:
        # Get higher timeframe trend
        higher_tf_trend = await self._get_higher_timeframe_trend(signal.symbol, signal.timeframe)
        
        # Check if signal aligns with higher timeframe
        if signal.signal_type == "buy" and higher_tf_trend == "bullish":
            return True
        elif signal.signal_type == "sell" and higher_tf_trend == "bearish":
            return True
        else:
            return False
            
    except Exception as e:
        logger.error(f"Timeframe confluence check failed: {e}")
        return False

async def _get_higher_timeframe_trend(self, symbol: str, current_tf: str) -> str:
    """Get higher timeframe trend direction"""
    # Map timeframes to higher ones
    tf_mapping = {
        "1m": "5m",
        "5m": "15m", 
        "15m": "1h",
        "1h": "4h",
        "4h": "1d",
        "1d": "1w"
    }
    
    higher_tf = tf_mapping.get(current_tf, "1d")
    
    # Get trend from higher timeframe
    # Implementation needed: Trend calculation
    return "bullish"  # Placeholder
```

## 📊 Signal Performance Tracking

### Signal History Model
```python
# Update: backend/database/models.py
class SignalHistory(Base):
    __tablename__ = "signal_history"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    signal_type = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    quality_score = Column(Float, nullable=False)
    executed = Column(Boolean, default=False)
    pnl = Column(Float, nullable=True)
    execution_time = Column(DateTime, nullable=True)
    
    # Performance metrics
    success = Column(Boolean, nullable=True)
    hold_time = Column(Interval, nullable=True)
    max_profit = Column(Float, nullable=True)
    max_loss = Column(Float, nullable=True)
```

### Performance Analytics
```python
class SignalPerformanceAnalyzer:
    def __init__(self):
        self.db_session = get_db_session()
    
    async def analyze_signal_performance(self, symbol: str = None, 
                                       timeframe: str = None) -> Dict:
        """Analyze signal performance"""
        try:
            # Get signal history
            query = self.db_session.query(SignalHistory)
            
            if symbol:
                query = query.filter(SignalHistory.symbol == symbol)
            if timeframe:
                query = query.filter(SignalHistory.timeframe == timeframe)
            
            signals = query.all()
            
            # Calculate metrics
            total_signals = len(signals)
            executed_signals = [s for s in signals if s.executed]
            successful_signals = [s for s in executed_signals if s.success]
            
            metrics = {
                "total_signals": total_signals,
                "execution_rate": len(executed_signals) / total_signals if total_signals > 0 else 0,
                "success_rate": len(successful_signals) / len(executed_signals) if executed_signals else 0,
                "avg_quality_score": sum(s.quality_score for s in signals) / total_signals if total_signals > 0 else 0,
                "avg_pnl": sum(s.pnl for s in executed_signals) / len(executed_signals) if executed_signals else 0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {}
```

## 🚀 Next Steps

1. **Complete webhook signal processing** pipeline
2. **Implement signal quality filtering** with volume confirmation
3. **Add multi-timeframe confluence** validation
4. **Set up signal performance tracking** and analytics
5. **Create advanced Pine Script** indicators for specific strategies

## 📚 Related Documentation

- [Data Collection Layer](./01_data_collection_layer.md)
- [Storage & Processing Layer](./02_storage_processing_layer.md)
- [Analysis Layer](./03_analysis_layer.md)
- [Execution Layer](./04_execution_layer.md)
- [Risk Management](./05_risk_management.md)
