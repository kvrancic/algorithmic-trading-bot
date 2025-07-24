# Getting Started with QuantumSentiment Trading Bot

## Prerequisites

### System Requirements
- Python 3.9 or higher
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space
- Stable internet connection
- macOS, Linux, or Windows 10/11

### Required Accounts
1. **Alpaca Markets Account**
   - Sign up at https://alpaca.markets
   - Get both paper and live trading API keys
   - Enable data subscription (free tier available)

2. **Reddit API Access**
   - Create app at https://www.reddit.com/prefs/apps
   - Note down client ID and secret

3. **News API Keys** (Optional but recommended)
   - NewsAPI.io account
   - Alpha Vantage API key
   - Benzinga API key (optional)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/algorithmic-trading-bot.git
cd algorithmic-trading-bot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Additional ML Libraries
```bash
# For GPU support (optional but recommended)
pip install tensorflow-gpu torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Configuration

### 1. Create Configuration File
```bash
cp config/config.example.yaml config/config.yaml
```

### 2. Set Up Environment Variables
Create a `.env` file in the project root:
```bash
# Alpaca API Keys
ALPACA_API_KEY_PAPER=your_paper_api_key
ALPACA_API_SECRET_PAPER=your_paper_api_secret
ALPACA_API_KEY_LIVE=your_live_api_key
ALPACA_API_SECRET_LIVE=your_live_api_secret

# Reddit API
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=QuantumSentimentBot/1.0

# News APIs
NEWS_API_KEY=your_newsapi_key
ALPHA_VANTAGE_KEY=your_alpha_vantage_key
BENZINGA_API_KEY=your_benzinga_key

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/trading_bot
```

### 3. Configure Trading Parameters
Edit `config/config.yaml`:
```yaml
trading:
  # Trading universe
  watchlist:
    - AAPL
    - GOOGL
    - MSFT
    - TSLA
    - SPY
  
  # Position sizing
  max_position_size: 0.1  # 10% max per position
  max_portfolio_positions: 10
  
  # Risk parameters
  stop_loss_pct: 0.02  # 2% stop loss
  take_profit_pct: 0.05  # 5% take profit
  max_daily_loss: 0.03  # 3% max daily loss

risk:
  max_drawdown: 0.10  # 10% maximum drawdown
  risk_per_trade: 0.01  # 1% risk per trade
  max_leverage: 1.0  # No leverage by default
```

## Database Setup

### 1. PostgreSQL Installation
```bash
# macOS
brew install postgresql
brew services start postgresql

# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql

# Create database
createdb trading_bot
```

### 2. Run Database Migrations
```bash
python scripts/setup_database.py
```

## Initial Setup and Testing

### 1. Verify Configuration
```bash
python scripts/verify_setup.py
```

This will check:
- API connectivity
- Database connection
- Configuration validity
- Required dependencies

### 2. Download Historical Data
```bash
python scripts/download_historical_data.py --symbols AAPL GOOGL MSFT --days 30
```

### 3. Run System Tests
```bash
pytest tests/
```

## Training Models (Optional)

If you want to train your own models instead of using pre-trained ones:

### 1. Prepare Training Data
```bash
python scripts/prepare_training_data.py --start-date 2023-01-01 --end-date 2024-01-01
```

### 2. Train Individual Models
```bash
# Train all models
python scripts/train_models.py --all

# Or train specific models
python scripts/train_models.py --model cnn
python scripts/train_models.py --model lstm
python scripts/train_models.py --model xgboost
python scripts/train_models.py --model transformer
```

### 3. Validate Models
```bash
python scripts/validate_models.py
```

## Running the Trading Bot

### 1. Paper Trading Mode (Recommended to start)
```bash
python -m src.main --mode paper --config config/config.yaml
```

### 2. Semi-Automatic Mode (Requires approval for trades)
```bash
python -m src.main --mode semi_auto --config config/config.yaml
```

### 3. Full Automatic Mode (Autonomous trading)
```bash
python -m src.main --mode full_auto --config config/config.yaml
```

### 4. Specific Symbol Trading
```bash
python -m src.main --mode paper --symbols AAPL GOOGL TSLA
```

## Monitoring the System

### 1. View Logs
```bash
# Real-time logs
tail -f logs/trading.log

# Error logs only
tail -f logs/errors.log
```

### 2. Performance Dashboard
Open another terminal:
```bash
python scripts/run_dashboard.py
```
Then open http://localhost:8050 in your browser

### 3. Check System Status
```bash
python scripts/check_status.py
```

## Common Operations

### 1. Emergency Stop
Press `Ctrl+C` to gracefully shut down the system. All pending orders will be cancelled.

### 2. Flatten All Positions
```bash
python scripts/flatten_all_positions.py
```

### 3. Export Performance Report
```bash
python scripts/generate_report.py --start-date 2024-01-01 --output reports/
```

### 4. Adjust Risk Parameters
Edit `config/config.yaml` and restart the bot. Changes take effect immediately.

## Troubleshooting

### Issue: "No module named 'src'"
**Solution**: Make sure you're running from the project root directory

### Issue: "API rate limit exceeded"
**Solution**: 
- Reduce the number of symbols in watchlist
- Increase the update intervals in config
- Check your Alpaca subscription tier

### Issue: "Database connection failed"
**Solution**:
- Verify PostgreSQL is running
- Check DATABASE_URL in .env file
- Run: `psql -U postgres -c "CREATE DATABASE trading_bot;"`

### Issue: "Insufficient buying power"
**Solution**:
- Check your account balance
- Reduce position sizes in config
- Ensure you're using the correct API keys (paper vs live)

### Issue: "Model prediction failed"
**Solution**:
- Verify models are properly trained/loaded
- Check for missing market data
- Review logs for specific error messages

## Best Practices

1. **Start with Paper Trading**: Always test strategies with paper trading first
2. **Monitor Regularly**: Check the system at least once per day
3. **Set Conservative Limits**: Start with small position sizes and tight risk controls
4. **Keep Logs**: Maintain detailed logs for analysis and debugging
5. **Regular Backups**: Backup your database and configuration regularly
6. **Update Dependencies**: Keep libraries updated for security and performance
7. **Review Performance**: Analyze trading results weekly/monthly

## Next Steps

1. Read the [System Architecture](system_architecture.md) guide
2. Review [Trading Strategies](trading_strategies.md) documentation
3. Learn about [Risk Management](risk_management.md)
4. Explore [Advanced Configuration](advanced_configuration.md)

## Support

- GitHub Issues: https://github.com/yourusername/algorithmic-trading-bot/issues