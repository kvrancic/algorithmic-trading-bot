# Trading Bot Control Guide

## 1. Download Historical Data

### Basic Download
```bash
# Download data for symbols in config.yaml
python -m src.data.download_historical_data

# Download specific symbols (override config)
python -m src.data.download_historical_data --symbols AAPL,TSLA,NVDA --days 365

# Download with specific timeframes
python -m src.data.download_historical_data --timeframes 1min,5min,1hour,1day
```

### Configuration Control
Edit `config/download_config.example.yaml` or create `config/download_config.yaml`:
```yaml
# Number of days to download
days: 730  # 2 years of data

# What to download
download_market_data: true
download_sentiment_data: true  
download_fundamental_data: true

# Symbols (or use config.yaml universe)
symbols: []  # Empty = use config.yaml universe
```

## 2. Train Your Models

### Quick Training
```bash
# Train all models with default config
python -m src.training.train_models

# Train specific model
python -m src.training.train_models --model price_lstm
python -m src.training.train_models --model sentiment_bert
python -m src.training.train_models --model ensemble
```

### Training Configuration
In `config/config.yaml`, control training:
```yaml
ml:
  models:
    price_lstm:
      enabled: true        # Enable/disable model
      sequence_length: 48  # 48 hours of data
      
    sentiment_bert:
      enabled: true        # Enable/disable model
      
    ensemble:
      voting: "weighted"   # How to combine models
      min_confidence: 0.65 # Only trade when confident
```

### Model Selection Control
```yaml
# Control which models contribute to trading decisions
trading:
  signal_sources:
    technical_only: false    # Use only technical analysis
    sentiment_only: false    # Use only sentiment analysis
    require_both: false      # Require both signals to agree
    ensemble_mode: true      # Use all available models (recommended)
```

## 3. Perform Trades

### Start Trading
```bash
# Paper trading (recommended first)
python -m src.main --mode paper

# Live trading (after testing)
python -m src.main --mode live
```

### Trading Configuration
```yaml
trading:
  # Position limits
  max_positions: 10
  max_position_size: 0.10    # 10% max per position
  min_position_size: 0.001   # 0.1% min per position
  
  # Signal thresholds
  signal_threshold: 0.7      # Minimum signal strength
  confidence_threshold: 0.6  # Minimum confidence to trade

risk:
  # Risk controls
  stop_loss_pct: 0.02       # 2% stop loss
  take_profit_pct: 0.05     # 5% take profit
  daily_loss_limit: 0.03    # 3% daily loss limit
  max_drawdown: 0.10        # 10% maximum drawdown
```

## 4. Monitor Trades & Performance

### Built-in Monitoring
The bot includes comprehensive monitoring:
- **Real-time dashboard**: http://localhost:8000 (when running)
- **Performance metrics**: Automatically tracked
- **Risk alerts**: Email/Discord/Telegram (if configured)

### Monitor via Alpaca Dashboard
- Log into [Alpaca](https://app.alpaca.markets/)
- View portfolio, positions, orders
- Check P&L and performance

### Export Performance Data
```bash
# Export trading performance
python -m src.analysis.export_performance --days 30

# Generate performance report
python -m src.analysis.generate_report --format pdf
```

### Monitoring Configuration
```yaml
monitoring:
  dashboard:
    enabled: true
    port: 8000
    
  alerts:
    telegram:
      enabled: false  # Set to true and add bot token
    email:
      enabled: false  # Set to true and add SMTP settings
```

## 5. Choose Trading Symbols

### Static Symbol Configuration
Edit `config/config.yaml`:
```yaml
universe:
  stocks: ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "SPY", "QQQ"]
  crypto: ["BTC-USD", "ETH-USD", "BNB-USD"] 
  forex: ["EUR/USD", "GBP/USD", "USD/JPY"]
  
  # Symbol filters
  filters:
    min_market_cap: 1e9    # $1B minimum market cap
    min_volume: 1e6        # $1M daily volume minimum
    max_spread: 0.01       # 1% max bid-ask spread
    
  # Dynamic discovery settings
  dynamic_discovery:
    enabled: true
    sentiment_threshold: 100    # Minimum mentions needed
    trending_window_hours: 24   # Look for trending symbols
    max_new_symbols_daily: 5    # Limit new additions
```

### Add New Symbols
```bash
# Add symbols temporarily
python -m src.main --mode paper --add-symbols NVDA,AMD,INTC

# Or edit config.yaml and restart
```

## 6. Dynamic Symbol Discovery

**Already implemented for you!** The system will:

1. **Monitor Sentiment Sources**: Reddit, news, social media
2. **Extract Symbol Mentions**: Automatically find trending stocks
3. **Validate Symbols**: Check market cap, volume, data availability
4. **Add to Universe**: Automatically include in trading (if enabled)

### Enable Dynamic Discovery
```yaml
universe:
  dynamic_discovery:
    enabled: true                    # Turn on auto-discovery
    sentiment_threshold: 100         # Min mentions to consider
    trending_window_hours: 24        # Time window for trending
    max_new_symbols_daily: 5         # Daily limit for new symbols
    confidence_threshold: 0.7        # Min confidence to add symbol
    
    # Source weights for discovery
    source_weights:
      reddit: 0.4          # Reddit mentions weight
      news: 0.3            # News mentions weight
      twitter: 0.2         # Twitter mentions weight
      unusual_whales: 0.1  # Options flow weight
```

## 7. Control Signal Sources

### Trading Strategy Selection
```yaml
trading:
  strategy_mode: "adaptive"  # adaptive, technical_only, sentiment_only, conservative
  
  signal_requirements:
    # Adaptive mode (recommended)
    adaptive:
      use_available_signals: true    # Use whatever signals are available
      confidence_boost_both: 0.1     # Extra confidence when both agree
      min_signal_strength: 0.6       # Minimum signal to trade
    
    # Technical only mode
    technical_only:
      enabled: false
      required_indicators: ["rsi", "macd", "bollinger"]
      min_confluence: 2              # How many indicators must agree
    
    # Sentiment only mode  
    sentiment_only:
      enabled: false
      required_sources: ["reddit", "news"]
      min_sentiment_score: 0.7       # Strong sentiment required
    
    # Conservative mode (both required)
    conservative:
      enabled: false
      require_technical_confirmation: true
      require_sentiment_confirmation: true
      higher_confidence_threshold: 0.8
```

### Real-time Strategy Switching
```bash
# Switch to technical-only mode
python -m src.trading.switch_mode --mode technical_only

# Switch to sentiment-only mode  
python -m src.trading.switch_mode --mode sentiment_only

# Switch back to adaptive (recommended)
python -m src.trading.switch_mode --mode adaptive
```

## Quick Start Checklist

1. **Setup**: Ensure `.env` file has API keys
2. **Configure**: Edit `config/config.yaml` with your preferences
3. **Download Data**: `python -m src.data.download_historical_data`
4. **Train Models**: `python -m src.training.train_models`
5. **Test Paper Trading**: `python -m src.main --mode paper`
6. **Monitor Dashboard**: Open http://localhost:8000
7. **Switch to Live**: `python -m src.main --mode live` (when ready)

## Emergency Controls

```bash
# Stop all trading immediately
python -m src.trading.emergency_stop

# Close all positions
python -m src.trading.close_all_positions

# Switch to paper mode
python -m src.trading.switch_to_paper
```

## Performance Optimization

```yaml
# Optimize for your trading style
trading:
  frequency: "intraday"      # intraday, daily, swing
  risk_tolerance: "medium"   # low, medium, high, aggressive
  
  # Intraday settings
  intraday:
    max_trades_per_day: 20
    min_hold_time_minutes: 15
    
  # Daily settings  
  daily:
    max_trades_per_day: 5
    min_hold_time_hours: 4
```