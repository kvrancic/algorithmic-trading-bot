# Risk Management Guide

## Overview

Risk management is the cornerstone of successful algorithmic trading. The QuantumSentiment Trading Bot implements multiple layers of risk controls to protect capital and ensure sustainable performance.

## Risk Management Philosophy

### Core Principles
1. **Capital Preservation First**: Never risk more than you can afford to lose
2. **Asymmetric Risk/Reward**: Target higher rewards than risks taken
3. **Diversification**: Spread risk across multiple positions and strategies
4. **Dynamic Adaptation**: Adjust risk based on market conditions
5. **Fail-Safe Mechanisms**: Multiple independent risk checks

## Risk Control Layers

### Layer 1: Pre-Trade Risk Checks

#### Signal Validation
```python
Before any trade:
1. Signal strength validation (threshold: 0.7)
2. Model confidence check (threshold: 0.6)
3. Risk budget availability
4. Correlation with existing positions
5. Market regime appropriateness
```

#### Position Sizing Constraints
```python
Size limits enforced:
- Max position size: 10% of portfolio
- Max sector concentration: 30%
- Max correlated exposure: 50%
- Min position size: $100 or 0.1% of portfolio
```

### Layer 2: Execution Risk Controls

#### Slippage Protection
```python
Slippage controls:
- Max acceptable slippage: 0.5%
- Adaptive execution based on liquidity
- Order type selection (limit vs market)
- Real-time order book monitoring
```

#### Market Impact Limits
```python
Impact thresholds:
- Max order size: 1% of average daily volume
- Kyle's Lambda estimation
- Participation rate limits
- Dark pool utilization when available
```

### Layer 3: Position-Level Risk Management

#### Stop Loss Implementation
```python
Stop loss types:

1. Fixed Stop Loss
   - Default: 2% below entry
   - Volatility-adjusted: 2 × ATR
   - Support-based placement

2. Trailing Stop Loss
   trigger_price = max_price × (1 - trailing_percent)
   where trailing_percent = 0.01 to 0.03

3. Time-Based Stop
   - Exit if position unprofitable after 24 hours
   - Gradual reduction: 50% after 48 hours
```

#### Take Profit Strategy
```python
Dynamic targets:
- Primary target: 5% or 3 × ATR
- Scaling out: 50% at first target
- Runner management: Trail remaining 50%
- Volatility adjustment: Wider in calm markets
```

### Layer 4: Portfolio-Level Risk Management

#### Maximum Drawdown Control
```python
Drawdown management:
- Warning level: 5% drawdown → Reduce position sizes
- Critical level: 8% drawdown → No new positions
- Emergency level: 10% drawdown → Flatten all positions

Implementation:
if drawdown > 0.05:
    position_size_multiplier = 0.5
if drawdown > 0.08:
    allow_new_positions = False
if drawdown > 0.10:
    emergency_liquidation()
```

#### Daily Loss Limits
```python
Daily risk budget:
- Soft limit: 2% daily loss → Reduce activity
- Hard limit: 3% daily loss → Stop trading

Reset: Next trading day at market open
```

#### Exposure Management
```python
Portfolio constraints:
- Max gross exposure: 100% (no leverage)
- Max net exposure: ±80%
- Min cash buffer: 10%
- Sector limits: 30% per sector
```

### Layer 5: System-Level Safeguards

#### Circuit Breakers
```python
Automatic shutdowns triggered by:
1. Technical failures
   - Data feed interruption > 60 seconds
   - Model prediction failures > 3 consecutive
   - Database connection loss

2. Market anomalies
   - Flash crash detection (5% move in 1 minute)
   - Liquidity crisis (spread > 1%)
   - Halt in primary positions

3. Account issues
   - Margin call warning
   - PDT violation risk
   - Compliance alerts
```

#### Risk Monitoring Dashboard
Real-time monitoring of:
- Current positions and P&L
- Risk metrics (VaR, CVaR, Sharpe)
- Exposure by sector/asset
- Correlation matrix
- Alert status

## Risk Metrics and Calculations

### Value at Risk (VaR)
```python
# 95% confidence interval, 1-day horizon
def calculate_var(returns, confidence=0.95):
    return np.percentile(returns, (1 - confidence) * 100)

# Expected: Daily VaR should not exceed 2% of portfolio
```

### Conditional VaR (CVaR)
```python
# Expected shortfall beyond VaR
def calculate_cvar(returns, confidence=0.95):
    var = calculate_var(returns, confidence)
    return returns[returns <= var].mean()
```

### Sharpe Ratio Monitoring
```python
# Risk-adjusted returns
def calculate_sharpe(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * excess_returns.mean() / returns.std()

# Target: Sharpe > 1.5
```

### Maximum Drawdown Tracking
```python
def calculate_max_drawdown(equity_curve):
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min()
```

## Risk Scenarios and Responses

### Scenario 1: Flash Crash
**Detection**: 5% market move in < 5 minutes
**Response**:
1. Immediate halt to new orders
2. Cancel all pending orders
3. Tighten stops on existing positions
4. Wait for market stabilization

### Scenario 2: Position Limit Breach
**Detection**: Position exceeds 10% of portfolio
**Response**:
1. Block additional buys
2. Generate alert for manual review
3. Create scaling plan
4. Execute gradual reduction

### Scenario 3: Correlation Spike
**Detection**: Portfolio correlation > 0.8
**Response**:
1. Identify correlated positions
2. Reduce largest positions
3. Seek uncorrelated opportunities
4. Implement hedging

### Scenario 4: Model Degradation
**Detection**: Prediction accuracy < 50% over 20 trades
**Response**:
1. Reduce position sizes by 50%
2. Increase confidence thresholds
3. Trigger model retraining
4. Consider strategy pause

## Risk Configuration

### Conservative Profile
```yaml
risk:
  max_position_size: 0.05  # 5%
  max_portfolio_positions: 20
  stop_loss_pct: 0.01  # 1%
  max_drawdown: 0.05  # 5%
  daily_loss_limit: 0.01  # 1%
  risk_per_trade: 0.005  # 0.5%
  
  # Strict correlation limits
  max_correlation: 0.5
  max_sector_exposure: 0.2  # 20%
```

### Moderate Profile
```yaml
risk:
  max_position_size: 0.10  # 10%
  max_portfolio_positions: 10
  stop_loss_pct: 0.02  # 2%
  max_drawdown: 0.10  # 10%
  daily_loss_limit: 0.03  # 3%
  risk_per_trade: 0.01  # 1%
  
  # Balanced limits
  max_correlation: 0.7
  max_sector_exposure: 0.3  # 30%
```

### Aggressive Profile
```yaml
risk:
  max_position_size: 0.15  # 15%
  max_portfolio_positions: 7
  stop_loss_pct: 0.03  # 3%
  max_drawdown: 0.15  # 15%
  daily_loss_limit: 0.05  # 5%
  risk_per_trade: 0.02  # 2%
  
  # Relaxed limits
  max_correlation: 0.8
  max_sector_exposure: 0.4  # 40%
```

## Risk Reporting

### Daily Risk Report
Generated automatically at market close:
```
Date: 2024-01-10
Portfolio Value: $100,000
Daily P&L: +$1,250 (+1.25%)

Risk Metrics:
- Current Drawdown: 2.3%
- Daily VaR (95%): $1,800
- Sharpe Ratio: 1.82
- Win Rate: 58%

Position Summary:
- Active Positions: 7
- Largest Position: AAPL (8.5%)
- Total Exposure: 67%
- Correlation: 0.45

Alerts: None
```

### Risk Analytics Dashboard
Real-time web interface showing:
- Portfolio heat map
- Risk gauge meters
- P&L attribution
- Correlation matrix
- Historical risk metrics

## Emergency Procedures

### 1. System Failure
```bash
# Emergency position flattening
python scripts/emergency_flatten.py --confirm

# Cancels all orders and closes all positions
```

### 2. Risk Limit Breach
```bash
# Check current risk status
python scripts/check_risk_status.py

# Manual risk reduction
python scripts/reduce_risk.py --target-exposure 0.5
```

### 3. Market Crisis
```bash
# Enable crisis mode
python scripts/crisis_mode.py --enable

# Implements:
# - Tighter stops
# - No new longs
# - Reduced position sizes
# - Increased cash buffer
```

## Best Practices

### 1. Regular Risk Reviews
- Daily: Check risk metrics and alerts
- Weekly: Review position correlations
- Monthly: Analyze risk-adjusted returns
- Quarterly: Adjust risk parameters

### 2. Stress Testing
```python
# Run monthly stress tests
scenarios = [
    "market_crash_20_percent",
    "flash_crash_recovery",
    "sector_rotation",
    "liquidity_crisis",
    "correlation_breakdown"
]

for scenario in scenarios:
    results = stress_test_portfolio(scenario)
    log_results(results)
```

### 3. Risk Education
- Understand each risk metric
- Know your risk tolerance
- Monitor market conditions
- Learn from losses
- Document risk events

### 4. Continuous Improvement
- Track risk metric effectiveness
- Refine stop loss placement
- Optimize position sizing
- Update risk models
- Learn from market events

## Risk Management Checklist

### Pre-Market
- [ ] Check account status and buying power
- [ ] Review overnight news and events
- [ ] Verify all systems operational
- [ ] Confirm risk parameters loaded

### During Trading
- [ ] Monitor position-level stops
- [ ] Track portfolio exposure
- [ ] Watch for correlation changes
- [ ] Check system health

### Post-Market
- [ ] Review daily P&L and risk metrics
- [ ] Analyze any risk events
- [ ] Update risk logs
- [ ] Plan next day's risk budget

### Weekly
- [ ] Comprehensive risk review
- [ ] Stress test portfolio
- [ ] Adjust parameters if needed
- [ ] Document lessons learned

Remember: Effective risk management is not about avoiding all risks, but about taking calculated risks with appropriate controls in place. The goal is sustainable, long-term profitability, not maximum short-term gains.