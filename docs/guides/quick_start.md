# Quick Start Guide - QuantumSentiment Trading Bot

## 5-Minute Setup

### 1. Prerequisites Check
```bash
# Check Python version (need 3.9+)
python --version

# Check pip
pip --version
```

### 2. Quick Install
```bash
# Clone and enter directory
git clone https://github.com/yourusername/algorithmic-trading-bot.git
cd algorithmic-trading-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Essential Configuration

Create `.env` file with your API keys:
```bash
# Minimum required for paper trading
ALPACA_API_KEY_PAPER=your_paper_key_here
ALPACA_API_SECRET_PAPER=your_paper_secret_here
DATABASE_URL=sqlite:///trading_bot.db  # Simple SQLite for quick start
```

### 4. Quick Config

Create `config/config.yaml`:
```yaml
# Minimal configuration for paper trading
mode: paper
database:
  connection_string: ${DATABASE_URL}

trading:
  watchlist:
    - AAPL
    - GOOGL
    - MSFT
  max_positions: 5
  max_position_size: 0.1

risk:
  max_drawdown: 0.10
  stop_loss_pct: 0.02
  risk_per_trade: 0.01
```

### 5. Start Trading!
```bash
# Run in paper trading mode
python -m src.main --mode paper
```

## First Day Trading Checklist

### Morning (Pre-Market)
1. **Start the Bot**
   ```bash
   python -m src.main --mode paper --symbols AAPL GOOGL MSFT
   ```

2. **Verify Connection**
   - Check logs: `tail -f logs/trading.log`
   - Should see: "Connected to Alpaca successfully"

3. **Monitor Initial Signals**
   - Bot will analyze market data
   - Generate predictions every 5 minutes
   - Look for "Signal generated" in logs

### During Market Hours
1. **Watch Your First Trade**
   - Signal appears: "Signal generated - AAPL BUY strength: 0.75"
   - Order placed: "Order submitted to Alpaca"
   - Position updated: "Position update added"

2. **Monitor Performance**
   ```bash
   # Check status
   python scripts/check_status.py
   ```

3. **View Positions**
   ```bash
   # See current positions
   python scripts/show_positions.py
   ```

### End of Day
1. **Check Results**
   ```bash
   # Generate summary
   python scripts/daily_summary.py
   ```

2. **Graceful Shutdown**
   - Press `Ctrl+C` to stop
   - Bot will cancel pending orders
   - Positions remain open for next day

## Common Scenarios

### Scenario 1: "I want to see what the bot would trade"
Use semi-auto mode:
```bash
python -m src.main --mode semi_auto
```
Bot will ask for approval before each trade.

### Scenario 2: "I want to trade specific stocks"
```bash
python -m src.main --mode paper --symbols TSLA NVDA AMD
```

### Scenario 3: "I want more conservative trading"
Edit `config/config.yaml`:
```yaml
trading:
  signal_threshold: 0.8  # Require stronger signals
  max_position_size: 0.05  # Smaller positions
risk:
  stop_loss_pct: 0.01  # Tighter stops
```

### Scenario 4: "I want to see the dashboard"
```bash
# In a new terminal
python scripts/run_dashboard.py
# Open http://localhost:8050
```

## Essential Commands

### Check Everything is Working
```bash
python scripts/verify_setup.py
```

### View Logs in Real-Time
```bash
tail -f logs/trading.log
```

### Emergency Stop
```bash
# Immediately flatten all positions
python scripts/emergency_flatten.py
```

### Daily Performance
```bash
python scripts/show_performance.py --days 1
```

## What to Expect

### First Hour
- Bot initializes and connects to Alpaca
- Downloads market data
- Starts analyzing sentiment (if configured)
- Begins generating predictions

### First Day
- Typically 0-5 trades (conservative settings)
- Small positions (5-10% each)
- Focus on liquid stocks
- Expect small gains/losses (<1%)

### First Week
- 10-30 total trades
- Learning your configured symbols
- Building performance history
- Refining predictions

## Quick Troubleshooting

### "No trades happening"
- Check market is open: `python scripts/check_market.py`
- Lower signal threshold in config
- Add more volatile symbols

### "Too many trades"
- Increase signal_threshold to 0.8
- Reduce watchlist size
- Add position limits

### "Connection errors"
- Verify API keys in .env
- Check internet connection
- Ensure market data subscription active

### "Bot crashed"
- Check logs: `tail -100 logs/trading.log`
- Restart: `python -m src.main --mode paper`
- Positions are safe (managed by broker)

## Next Steps

1. **Run for a full week in paper mode**
   - Monitor daily performance
   - Understand the bot's behavior
   - Note any issues

2. **Customize your strategy**
   - Adjust risk parameters
   - Add/remove symbols
   - Tune signal thresholds

3. **Learn the system**
   - Read [Trading Strategies](trading_strategies.md)
   - Understand [Risk Management](risk_management.md)
   - Explore [System Architecture](system_architecture.md)

4. **Join the community**
   - Share your results
   - Get configuration tips
   - Learn from others

Remember: Start small, monitor closely, and gradually increase complexity as you gain confidence with the system.

## Safety Reminder

⚠️ **IMPORTANT**: 
- Always start with paper trading
- Never trade money you can't afford to lose
- Monitor the bot regularly
- Understand the risks involved
- This is not financial advice

Happy Trading! 🚀