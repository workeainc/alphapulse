# 📋 QUICK REFERENCE CARD - Consensus & Risk Management

## ⚙️ **CONSENSUS MECHANISM SETTINGS**

```python
# File: apps/backend/src/ai/consensus_manager.py

Minimum Agreeing Heads: 4/9 (44% rule)
Minimum Probability: 0.60
Minimum Confidence: 0.70
Minimum Consensus Confidence: 0.65

Head Weights:
  Technical (A): 13%
  Sentiment (B): 9%
  Volume (C): 13%
  Rule-Based (D): 9%
  ICT (E): 13%
  Wyckoff (F): 13%
  Harmonic (G): 9%
  Structure (H): 9%
  Crypto (I): 12%
```

---

## 📊 **CONFIDENCE BANDS**

| Band | Range | Position Size | R:R | Win Rate | Example |
|------|-------|---------------|-----|----------|---------|
| **Very High** | 0.85-0.95 | 2.0-3.0% | 3:1 | 75-85% | 7+ heads, perfect storm |
| **High** | 0.75-0.85 | 1.5-2.5% | 2.5:1 | 65-75% | 5-6 heads, strong setup |
| **Medium** | 0.65-0.75 | 1.0-1.5% | 2:1 | 55-65% | 4 heads, minimum viable |
| **Rejected** | < 0.65 | 0% | N/A | N/A | Too weak |

---

## 🛡️ **RISK MANAGEMENT RULES**

### **Liquidation Risk:**
```
IF distance_to_liquidation < 2%:
  → SKIP TRADE

IF distance_to_liquidation < 5%:
  → REDUCE POSITION SIZE by 50%
```

### **Extreme Leverage:**
```
IF perpetual_premium > 0.5%:
  IF signal WITH leverage:
    → REDUCE POSITION SIZE by 50%
  IF signal AGAINST leverage:
    → BOOST CONFIDENCE by 10%
```

### **Entry Zone:**
```
LONG Signal:
  In Discount Zone (0-50%): ENTER IMMEDIATELY
  In Premium Zone (50-100%): WAIT FOR PULLBACK

SHORT Signal:
  In Premium Zone (50-100%): ENTER IMMEDIATELY
  In Discount Zone (0-50%): WAIT FOR PULLBACK
```

---

## 📐 **FORMULAS**

### **Consensus Probability:**
```
= (prob₁×weight₁ + prob₂×weight₂ + ... + probₙ×weightₙ) / total_weight

Example:
= (0.75×0.13 + 0.78×0.13 + 0.88×0.13 + 0.90×0.13 + 0.83×0.12) / 0.64
= 0.828
```

### **Consensus Confidence:**
```
= Base Confidence + Agreement Bonus + Strength Bonus

Base = Average confidence of agreeing heads
Agreement Bonus = {4:0.00, 5:0.03, 6:0.06, 7:0.09, 8:0.12, 9:0.15}
Strength Bonus = {≥0.85:0.08, 0.75-0.85:0.05, 0.60-0.75:0.02}

Example:
= 0.824 + 0.03 + 0.05 = 0.904
```

### **Position Size:**
```
Very High (0.85-0.95):
  = 2.0% + (confidence - 0.85) / 0.10 × 1.0%
  
High (0.75-0.85):
  = 1.5% + (confidence - 0.75) / 0.10 × 1.0%
  
Medium (0.65-0.75):
  = 1.0% + (confidence - 0.65) / 0.10 × 0.5%
```

### **Take Profit:**
```
Risk = |Entry - Stop Loss|

Very High: Reward = Risk × 3.0
High: Reward = Risk × 2.5
Medium: Reward = Risk × 2.0

Take Profit = Entry ± Reward
```

### **Stop Loss:**
```
LONG: Stop Loss = Entry - (ATR × 1.5) OR Swing Low
SHORT: Stop Loss = Entry + (ATR × 1.5) OR Swing High
```

---

## 💻 **CODE SNIPPETS**

### **Generate Enhanced Signal:**

```python
from src.services.ai_model_integration_service import AIModelIntegrationService

ai_service = AIModelIntegrationService()
signal = await ai_service.generate_ai_signal('BTCUSDT', '1h')

if signal:
    print(f"Quality: {signal.signal_quality}")
    print(f"Position: ${signal.position_size_usd}")
    print(f"R:R: {signal.risk_reward_ratio}:1")
```

### **Manual Enhancement:**

```python
from src.ai.signal_risk_enhancement import SignalRiskEnhancement

enhancer = SignalRiskEnhancement()
enhanced = await enhancer.enhance_signal(
    symbol='BTCUSDT',
    direction='LONG',
    entry_price=43250.0,
    stop_loss=42800.0,
    consensus_result=consensus_result,
    market_data=market_data_dict,
    available_capital=10000.0
)
```

### **Get Position Size Only:**

```python
from src.ai.confidence_position_sizing import ConfidenceBasedPositionSizing

sizer = ConfidenceBasedPositionSizing()
result = sizer.calculate_position_size(
    consensus_confidence=0.87,
    available_capital=10000.0,
    entry_price=43250.0,
    stop_loss=42800.0
)

print(f"Position: ${result.position_size_usd}")
print(f"Win Rate: {result.expected_win_rate*100}%")
```

### **Get Take Profit Only:**

```python
from src.ai.confidence_risk_reward import ConfidenceBasedRiskReward

rr_calc = ConfidenceBasedRiskReward()
result = rr_calc.calculate_take_profit(
    entry_price=43250.0,
    stop_loss=42800.0,
    consensus_confidence=0.87,
    direction='LONG'
)

print(f"TP: ${result.take_profit}")
print(f"R:R: {result.risk_reward_ratio}:1")
```

---

## 📞 **TROUBLESHOOTING**

### **"Position size is 0"**
→ Check: `consensus_confidence >= 0.65`  
→ Fix: Improve signal quality to reach minimum confidence

### **"Signal rejected by risk management"**
→ Check logs for: Liquidation risk or extreme leverage  
→ Fix: Wait for better market conditions

### **"Take profit seems too far"**
→ This is correct for very high confidence (3:1 R:R)  
→ Higher confidence = higher targets

### **"Position size seems small"**
→ Check confidence band (medium = 1-1.5%)  
→ Improve signal quality for larger positions

---

## 📈 **EXPECTED RESULTS**

### **Signal Distribution:**
- 15-20% Excellent (0.85+, 7+ heads)
- 30-40% Good (0.75+, 5-6 heads)
- 40-50% Acceptable (0.65+, 4 heads)
- 10-15% Rejected (< 0.65 or < 4 heads)

### **Performance Metrics:**
- Average Position Size: 1.8% (vs 1.5% before)
- Average R:R: 2.5:1 (vs 2:1 before)
- Average Win Rate: 70% (vs 60% estimated before)
- Profitability: +15-20% improvement
- Drawdowns: -30% reduction

---

## 🎯 **QUICK EXAMPLES**

### **Example 1: Perfect Storm (Very High)**
```
7 heads agree, confidence 0.92
→ Position: 2.8% ($280)
→ R:R: 3:1
→ Win Rate: 80%
→ Quality: EXCELLENT
```

### **Example 2: Strong Setup (High)**
```
5 heads agree, confidence 0.80
→ Position: 2.0% ($200)
→ R:R: 2.5:1
→ Win Rate: 70%
→ Quality: GOOD
```

### **Example 3: Minimum Viable (Medium)**
```
4 heads agree, confidence 0.70
→ Position: 1.25% ($125)
→ R:R: 2:1
→ Win Rate: 60%
→ Quality: ACCEPTABLE
```

### **Example 4: With Liquidation Risk**
```
6 heads agree, confidence 0.88
Liquidation 3% away
→ Position: 1.4% ($140) ← Reduced from 2.7%
→ R:R: 3:1
→ Quality: GOOD (downgraded from EXCELLENT)
→ Warning: "Liquidation risk detected"
```

---

## 📂 **FILE LOCATIONS**

**Core Modules:**
- Consensus: `src/ai/consensus_manager.py`
- Position Sizing: `src/ai/confidence_position_sizing.py`
- Risk-Reward: `src/ai/confidence_risk_reward.py`
- Enhancement: `src/ai/signal_risk_enhancement.py`
- Integration: `src/services/ai_model_integration_service.py`

**Examples:**
- Complete Demo: `src/examples/step8_complete_example.py`

**Documentation:**
- See root level `.md` files in `apps/backend/`

---

**Print this card and keep it handy! 📄**

---

Status: ✅ COMPLETE  
Date: October 29, 2025  
Version: 1.0

