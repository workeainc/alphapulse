# Quick Integration Guide - Smart Signal Generator

## 🚀 5-Minute Integration

### **Step 1: Import**

```python
from apps.backend.src.ai.smart_signal_generator import SmartSignalGenerator
```

### **Step 2: Initialize (Once at Startup)**

```python
# Simple (use defaults)
generator = SmartSignalGenerator()

# OR with custom config
generator = SmartSignalGenerator(config={
    'adaptive': {
        'target_min_signals': 3,  # Min signals/day
        'target_max_signals': 8,   # Max signals/day
        'adjustment_interval_hours': 6  # Auto-adjust frequency
    }
})
```

### **Step 3: Generate Signals**

```python
result = await generator.generate_signal(
    symbol='BTC/USDT',
    timeframe='1h',
    market_data={
        'current_price': 50000.0,
        'volume': 1000000.0,
        'indicators': {
            'sma_20': 49500.0,
            'sma_50': 48000.0,
            'rsi_14': 65.0,
            'macd': 150.0,
            # Add more indicators here
        }
    },
    analysis_results={
        'dataframe': df,  # Your OHLCV DataFrame
        'volume_analysis': {...},
        'sentiment_analysis': {...}
    }
)
```

### **Step 4: Use Result**

```python
if result and result.signal_generated:
    # Execute trade
    execute_trade(
        symbol=result.symbol,
        direction=result.direction,
        confidence=result.confidence,
        quality=result.quality_score
    )
```

---

## 📊 Key Fields

### **SmartSignalResult:**

```python
result.signal_generated     # bool: Was signal generated?
result.symbol              # str: 'BTC/USDT'
result.timeframe           # str: '1h', '4h', etc.
result.direction           # str: 'long', 'short', 'flat'
result.confidence          # float: 0.70-0.90 (adjusted)
result.quality_score       # float: 0.65-0.85
result.consensus_score     # float: Raw consensus score
result.market_regime       # str: 'trending', 'ranging', etc.
result.contributing_heads  # list: ['technical', 'volume', ...]
result.reasoning           # str: Human-readable explanation
result.timestamp           # datetime: When generated
result.metadata            # dict: Full details
```

---

## 🎯 Common Patterns

### **Pattern 1: Basic Signal Loop**

```python
# Scan multiple pairs
for symbol in ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']:
    result = await generator.generate_signal(
        symbol=symbol,
        timeframe='1h',
        market_data=get_market_data(symbol),
        analysis_results=get_analysis(symbol)
    )
    
    if result and result.signal_generated:
        await send_signal_to_users(result)
```

### **Pattern 2: Confidence-Based Position Sizing**

```python
if result and result.signal_generated:
    # Higher confidence = Larger position
    position_multiplier = result.confidence
    position_size = base_size * position_multiplier
    
    execute_trade(
        symbol=result.symbol,
        direction=result.direction,
        size=position_size
    )
```

### **Pattern 3: Quality Filtering**

```python
if result and result.signal_generated:
    # Only take highest quality signals
    if result.quality_score >= 0.80:
        execute_trade(result)
    else:
        log_signal_skipped(result, reason="Quality too low")
```

### **Pattern 4: Regime-Aware Trading**

```python
if result and result.signal_generated:
    # Adjust strategy based on regime
    if result.market_regime == 'trending':
        use_trend_following_strategy(result)
    elif result.market_regime == 'ranging':
        use_mean_reversion_strategy(result)
    elif result.market_regime == 'volatile':
        reduce_position_size(result)
```

---

## 📈 Monitoring

### **Get Statistics:**

```python
stats = generator.get_stats()

print(f"Signals Today: {stats['signals_last_24h']}")
print(f"Avg Confidence: {stats['avg_confidence']:.3f}")
print(f"Avg Quality: {stats['avg_quality_score']:.3f}")
print(f"Signal Rate: {stats['signal_rate']:.2%}")
print(f"Duplicates Blocked: {stats['signals_blocked_duplicate']}")
```

### **Check Adaptive Thresholds:**

```python
thresholds = generator.adaptive_controller.get_current_thresholds()

print(f"Current Confidence Threshold: {thresholds['min_confidence']:.3f}")
print(f"Current Consensus Heads: {thresholds['min_consensus_heads']}")
print(f"Duplicate Window: {thresholds['duplicate_window_hours']}h")
```

---

## ⚙️ Configuration Options

### **Adaptive Controller:**

```python
config = {
    'adaptive': {
        'target_min_signals': 3,           # Min signals/day (default: 3)
        'target_max_signals': 8,            # Max signals/day (default: 8)
        'adjustment_interval_hours': 6,     # Auto-adjust frequency (default: 6)
    }
}
```

### **Quality Thresholds:**

These are **automatically adjusted** by the system, but you can see current values:

```python
# These auto-adjust every 6 hours
controller.min_confidence          # 0.70 - 0.90 (adaptive)
controller.min_consensus_heads     # 3 - 6 (adaptive)
controller.min_quality_score       # 0.65 - 0.85 (adaptive)
controller.duplicate_window_hours  # 2 - 8 (adaptive)
```

---

## 🎯 Integration Checklist

- [ ] Import `SmartSignalGenerator`
- [ ] Initialize once at startup
- [ ] Prepare market_data dict with indicators
- [ ] Prepare analysis_results dict with DataFrame
- [ ] Call `generate_signal()` for each pair/timeframe
- [ ] Check `result.signal_generated`
- [ ] Use `result.confidence` and `result.quality_score`
- [ ] Monitor statistics periodically
- [ ] (Optional) Log detailed reasoning from `result.reasoning`

---

## 🚨 Common Pitfalls

### **❌ DON'T:**
```python
# Don't create new generator for each signal
for symbol in symbols:
    generator = SmartSignalGenerator()  # ❌ BAD - recreates state
    result = await generator.generate_signal(...)
```

### **✅ DO:**
```python
# Create once, reuse many times
generator = SmartSignalGenerator()  # ✅ GOOD - reuses state

for symbol in symbols:
    result = await generator.generate_signal(...)
```

---

### **❌ DON'T:**
```python
# Don't ignore the quality_score
if result and result.signal_generated:
    execute_trade(result)  # ❌ BAD - might be low quality
```

### **✅ DO:**
```python
# Always check quality
if result and result.signal_generated and result.quality_score >= 0.70:
    execute_trade(result)  # ✅ GOOD - quality-filtered
```

---

## 💡 Pro Tips

1. **Let It Adapt:** Don't manually override thresholds - the system auto-tunes every 6 hours
2. **Monitor Stats:** Check `generator.get_stats()` daily to see performance
3. **Use Quality Score:** Filter on `quality_score >= 0.75` for highest-quality signals only
4. **Respect Regime:** Consider `result.market_regime` in your execution strategy
5. **Log Everything:** Save `result.metadata` for post-trade analysis

---

## 🎉 That's It!

You're ready to use the Smart Signal Generator. It will:
- ✅ Auto-calculate 50+ indicators
- ✅ Auto-aggregate intelligently
- ✅ Auto-adjust thresholds
- ✅ Auto-detect market regime
- ✅ Auto-prevent duplicates
- ✅ Auto-maintain 3-8 signals/day

**Just call `generate_signal()` and let the intelligence work for you!** 🚀

---

## 📚 More Info

- **Full Documentation:** `docs/SMART_TIERED_INTELLIGENCE_IMPLEMENTATION.md`
- **Technical Details:** `docs/SMART_ARCHITECTURE_FINAL_SUMMARY.md`
- **Example Script:** `apps/backend/examples/smart_signal_generator_usage.py`

---

**Need Help?**
- Check the example script for complete walkthrough
- Read the metadata field for detailed reasoning
- Monitor statistics to understand system behavior

**Happy Trading! 📈**

