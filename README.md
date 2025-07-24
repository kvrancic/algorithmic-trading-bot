# README.md

# QuantumSentiment: Institutional-Grade AI Trading System

> "Treat your €1,000 like a lab rat: valuable for science, expendable for ethics."

## ⚠️ RISK & COMPLIANCE DISCLAIMER

**CAPITAL AT RISK:** This system employs advanced quantitative strategies that can amplify both gains and losses. Your entire €1,000 can evaporate faster than water on Mars. No amount of mathematics can eliminate market risk.

**REGULATORY NOTICE:** Algorithmic trading is regulated in most jurisdictions. You are responsible for compliance with local laws, tax reporting, and any licensing requirements. Some strategies (like pattern day trading) may require minimum capital levels.

**NOT FINANCIAL ADVICE:** This software is for educational and research purposes. Past backtested performance is not indicative of future results. The developers assume no liability for financial losses.

## 🔧 Requirements

- **OS:** Ubuntu 20.04+ / macOS 12+ (Windows WSL2 possible but not recommended)
- **Python:** 3.10+ with scientific stack
- **RAM:** 8GB minimum (16GB recommended for ML models)
- **GPU:** Optional but recommended for deep learning models (CUDA 11.0+)
- **Storage:** 50GB+ for historical data and model checkpoints
- **Broker:** Alpaca account (free, paper trading available)
- **Data APIs:** Reddit account, Alpha Vantage key (free tier)
- **Dependencies:** See `requirements.txt` for full list

## 🏗️ Architecture

**Core Components:**
- **Data Pipeline:** MCP servers → Feature extraction → Time series database
- **Alpha Generation:** Multiple independent strategies with ensemble voting
- **ML Models:** LSTM price prediction, XGBoost pattern recognition, Transformer sentiment
- **Portfolio Management:** Markowitz optimization with Black-Litterman views
- **Risk Management:** VaR, Kelly Criterion, dynamic position sizing
- **Execution Engine:** Smart order routing with slippage modeling

**System Architecture:**
```
┌─────────────────────────┐
│   MCP Data Servers      │
├─────────────────────────┤
│ • Alpaca (Stocks/Crypto)│
│ • DexPaprika (DeFi)     │
│ • WSB Analyst (Reddit)  │
│ • Alpha Vantage (Macro) │
└───────────┬─────────────┘
            │
     ┌──────▼──────┐
     │ Feature     │      ┌─────────────────┐
     │ Engineering │      │ ML Model Zoo    │
     │ Pipeline    │      ├─────────────────┤
     └──────┬──────┘      │ • Price LSTM    │
            │             │ • Pattern CNN   │
     ┌──────▼──────┐      │ • Sentiment BERT│
     │ Signal      │◄─────┤ • XGBoost Trees │
     │ Generation  │      │ • Order Book NN │
     └──────┬──────┘      └─────────────────┘
            │
     ┌──────▼──────┐
     │ Portfolio   │      ┌─────────────────┐
     │ Optimizer   │◄─────┤ Risk Models     │
     └──────┬──────┘      │ • VaR/CVaR      │
            │             │ • Kelly Sizing  │
     ┌──────▼──────┐      │ • Drawdown Ctrl │
     │ Execution   │      └─────────────────┘
     │ Engine      │
     └──────┬──────┘
            │
     ┌──────▼──────┐
     │ Alpaca MCP  │
     │ (Orders)    │
     └─────────────┘
```

## 🚀 Setup Steps

### 1. Clone and Environment Setup
```bash
# Clone repository
git clone https://github.com/yourusername/quantumsentiment.git
cd quantumsentiment

# Create conda environment (recommended for scientific packages)
conda create -n quantum python=3.10
conda activate quantum

# Install core dependencies
pip install -r requirements.txt

# Install ML dependencies
pip install -r requirements-ml.txt

# Optional: Install GPU support
pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

### 2. Install MCP Servers
```bash
# Run automated MCP setup
./scripts/install_mcp_servers.sh

# Or manually install each:
# Alpaca MCP
cd mcp_servers
git clone https://github.com/laukikk/alpaca-mcp.git
cd alpaca-mcp && pip install -e . && cd ../..

# DexPaprika MCP (no auth needed!)
npm install -g @coinpaprika/dexpaprika-mcp

# WSB Analyst MCP
git clone https://github.com/ferdousbhai/wsb-analyst-mcp.git
cd wsb-analyst-mcp && npm install && cd ../..
```

### 3. Configure Environment
```bash
# Copy and edit configuration
cp config/config.example.yaml config/config.yaml
cp .env.example .env

# Edit with your credentials
nano .env

# Validate configuration
python scripts/validate_config.py
```

### 4. Download Historical Data
```bash
# Fetch historical data for backtesting (this may take 30-60 minutes)
python scripts/download_historical_data.py \
  --start-date 2022-01-01 \
  --end-date 2024-12-31 \
  --symbols "BTC,ETH,SPY,QQQ,TSLA,GME" \
  --resolution "1h"
```

### 5. Train Initial Models
```bash
# Train base models on historical data
python scripts/train_models.py --config config/models.yaml

# This will train:
# - LSTM for price prediction
# - CNN for chart pattern recognition  
# - XGBoost for technical indicators
# - BERT for sentiment analysis
```

### 6. Run Comprehensive Tests
```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Backtesting validation
python scripts/backtest.py --strategy all --validate

# Risk limit tests
python scripts/test_risk_limits.py --capital 1000
```

### 7. Launch System (Paper Mode)
```bash
# Start all MCP servers
./scripts/start_mcp_servers.sh

# Launch trading system in paper mode
python src/main.py --mode paper --config config/config.yaml

# Monitor in real-time
python src/monitor.py --dashboard
```

### 8. Analyze Performance
```bash
# Generate performance report after running
python scripts/generate_report.py --output reports/

# View in browser
open reports/performance_analysis.html
```

### 9. Go Live (⚠️ REAL MONEY ⚠️)
```bash
# Only after successful paper trading!
# Update config/config.yaml: trading_mode: "live"
# Update .env: CONFIRM_REAL_MONEY_TRADING=true

# Final safety check
python scripts/preflight_check.py --mode live

# Launch with real capital
python src/main.py --mode live --config config/config.yaml
```

## 📊 Advanced Strategy Logic

### 1. Feature Engineering Pipeline (100+ features)
```python
# Technical Features
- Price Action: OHLCV, returns, log returns, volatility
- Moving Averages: SMA, EMA, WMA (multiple timeframes)  
- Momentum: RSI, MACD, Stochastic, Williams %R
- Volatility: ATR, Bollinger Bands, Keltner Channels
- Volume: OBV, Volume Profile, VWAP
- Market Structure: Support/Resistance, Fibonacci levels

# Microstructure Features  
- Order Book: Bid-ask spread, depth imbalance, order flow
- Trade Flow: Buy/sell pressure, large trade detection
- Liquidity: Market impact estimation, slippage prediction

# Sentiment Features
- Reddit: Post velocity, comment sentiment, award ratios
- News: Entity sentiment, event detection, topic modeling
- Social: Influencer activity, viral coefficient

# Market Regime Features
- Correlation matrices, sector rotation signals
- VIX/Fear indices, term structure
- Macro indicators: yields, dollar index
```

### 2. Machine Learning Ensemble
```python
# Price Prediction Models
lstm_model = PriceLSTM(
    features=100,
    sequence_length=48,  # 48 hours lookback
    forecast_horizon=24  # 24 hours ahead
)

# Pattern Recognition
pattern_cnn = ChartPatternCNN(
    patterns=['head_shoulders', 'triangle', 'flag', 'wedge'],
    confidence_threshold=0.8
)

# Market Regime Classification  
regime_model = MarketRegimeXGBoost(
    regimes=['trending', 'ranging', 'volatile'],
    features=['volatility', 'volume', 'correlation']
)

# Sentiment Analysis
sentiment_transformer = FinBERT(
    fine_tuned_on=['wsb_posts', 'crypto_twitter', 'financial_news']
)

# Meta-Learning Ensemble
ensemble = StackedEnsemble(
    models=[lstm_model, pattern_cnn, regime_model, sentiment_transformer],
    meta_learner=LightGBM(),
    voting='weighted'  # Weights based on recent performance
)
```

### 3. Portfolio Optimization
```python
# Modern Portfolio Theory with ML predictions
optimizer = BlackLittermanOptimizer(
    market_implied_returns=True,
    ml_views=ensemble.predictions,
    confidence_scaling=True
)

# Risk Parity Allocation
risk_parity = RiskParityAllocator(
    target_volatility=0.15,  # 15% annual vol target
    rebalance_frequency='daily'
)

# Kelly Criterion with safety margin
kelly = FractionalKelly(
    fraction=0.25,  # Conservative 1/4 Kelly
    max_position_size=0.10,  # 10% max per position
    correlation_penalty=True
)
```

### 4. Advanced Risk Management
```python
# Value at Risk (VaR) and Conditional VaR
risk_engine = RiskEngine(
    var_confidence=0.95,
    cvar_threshold=0.99,
    max_portfolio_var=0.05  # 5% daily VaR limit
)

# Dynamic Stop Loss using ATR and volatility regime
stop_loss = AdaptiveStopLoss(
    base_atr_multiplier=2.0,
    volatility_adjustment=True,
    regime_specific={
        'trending': 3.0,
        'ranging': 1.5,
        'volatile': 1.0
    }
)

# Maximum Drawdown Control
drawdown_controller = DrawdownController(
    max_drawdown=0.20,  # 20% maximum
    reduction_factor=0.5,  # Cut position sizes by 50% near limit
    recovery_threshold=0.10  # Resume full size after 10% recovery
)
```

### 5. Execution Algorithms
```python
# Smart Order Routing
router = SmartOrderRouter(
    strategies=['TWAP', 'VWAP', 'Iceberg'],
    slippage_model=XGBoostSlippagePredictor(),
    urgency_scoring=True
)

# Market Impact Minimization
impact_minimizer = MarketImpactMinimizer(
    model='Almgren-Chriss',
    max_participation_rate=0.10,  # Max 10% of volume
    adaptive_scheduling=True
)
```

## 🎯 Advanced Sentiment Analysis Engine

### Multi-Source Sentiment Architecture
```python
class SentimentAggregator:
    """
    Combines signals from Reddit, Twitter, UnusualWhales, and custom sources
    """
    def __init__(self):
        self.sources = {
            'reddit': RedditSentimentAnalyzer(
                subreddits=['wallstreetbets', 'stocks', 'cryptocurrency'],
                lookback_hours=24,
                min_score_threshold=100
            ),
            'twitter': TwitterSentimentAnalyzer(
                accounts=['whale_alert', 'zerohedge', 'jimcramer'],
                hashtags=['StockMarket', 'Crypto', 'Trading'],
                enabled=bool(os.getenv('TWITTER_BEARER_TOKEN'))
            ),
            'unusual_whales': UnusualWhalesAnalyzer(
                endpoints=['politics', 'flow', 'darkpool'],
                scrape_frequency='15min'
            ),
            'news': NewsAggregator(
                sources=['bloomberg', 'reuters', 'coindesk'],
                entity_extraction=True
            )
        }
```

### Reddit Deep Analysis
```python
class RedditSentimentAnalyzer:
    """
    Advanced Reddit sentiment extraction with meme detection
    """
    def analyze_wsb(self, ticker):
        features = {
            'mention_velocity': self.calculate_mention_acceleration(ticker),
            'sentiment_score': self.bert_sentiment(ticker),
            'rocket_ratio': self.count_emoji_signals(ticker, ['🚀', '🌙', '💎']),
            'bear_signals': self.count_emoji_signals(ticker, ['🌈', '🐻', '📉']),
            'dd_quality': self.analyze_dd_posts(ticker),
            'comment_momentum': self.measure_comment_velocity(ticker),
            'award_weight': self.calculate_award_significance(ticker),
            'user_credibility': self.aggregate_user_karma(ticker),
            'option_flow': self.extract_strike_mentions(ticker),
            'gain_porn_correlation': self.correlate_with_gains(ticker)
        }
        return self.wsb_signal_model.predict(features)
```

### Twitter Sentiment with Optional API
```python
class TwitterSentimentAnalyzer:
    """
    Fallback gracefully if no Twitter API key provided
    """
    def __init__(self, enabled=True):
        self.enabled = enabled
        if enabled and not os.getenv('TWITTER_BEARER_TOKEN'):
            logger.warning("Twitter API key not found, disabling Twitter sentiment")
            self.enabled = False
            
    def get_sentiment(self, symbol):
        if not self.enabled:
            return {'signal': 0, 'confidence': 0, 'source': 'twitter_disabled'}
            
        # Advanced Twitter analysis
        tweets = self.fetch_tweets(symbol)
        features = {
            'influencer_sentiment': self.analyze_influencers(tweets),
            'retail_sentiment': self.analyze_retail(tweets),
            'velocity': self.calculate_tweet_acceleration(tweets),
            'reach': self.estimate_total_reach(tweets),
            'bot_ratio': self.detect_bot_activity(tweets)
        }
        return self.twitter_model.predict(features)
```

### UnusualWhales Political Intelligence
```python
class UnusualWhalesAnalyzer:
    """
    Scrapes UnusualWhales for political trading signals
    FREE tier: Web scraping with respectful rate limits
    PAID tier: Direct API access if available
    """
    def __init__(self):
        self.scraper = CloudflareScraper()  # Handles CF protection
        self.cache = RedisCache(ttl=300)  # 5 min cache
        
    async def get_political_trades(self):
        # Check cache first
        cached = self.cache.get('uw_political_trades')
        if cached:
            return cached
            
        # Scrape with rate limiting
        async with self.rate_limiter(calls=1, period=60):  # 1 req/min
            html = await self.scraper.get('https://unusualwhales.com/politics')
            trades = self.parse_political_trades(html)
            
            # Extract key signals
            signals = {
                'congress_buying': self.analyze_congress_positions(trades),
                'sector_rotation': self.detect_political_sector_bias(trades),
                'insider_confidence': self.calculate_politician_conviction(trades),
                'party_divergence': self.measure_party_trading_diff(trades),
                'timing_alpha': self.analyze_trade_timing_vs_news(trades)
            }
            
            self.cache.set('uw_political_trades', signals)
            return signals
    
    def parse_political_trades(self, html):
        """Extract trades with BeautifulSoup"""
        soup = BeautifulSoup(html, 'html.parser')
        trades = []
        
        for row in soup.select('.trade-row'):
            trade = {
                'politician': row.select_one('.politician-name').text,
                'ticker': row.select_one('.ticker').text,
                'transaction': row.select_one('.transaction-type').text,
                'amount': self.parse_amount(row.select_one('.amount').text),
                'date': self.parse_date(row.select_one('.file-date').text),
                'party': row.select_one('.party').text
            }
            trades.append(trade)
            
        return trades
```

### Sentiment Fusion Algorithm
```python
class SentimentFusion:
    """
    Combines all sentiment sources with adaptive weighting
    """
    def __init__(self):
        self.weights = {
            'reddit': 0.30,
            'twitter': 0.20,
            'unusual_whales': 0.35,  # Higher weight for insider info
            'news': 0.15
        }
        self.confidence_threshold = 0.65
        
    def fuse_signals(self, symbol):
        signals = {}
        
        # Gather from all sources
        for source, analyzer in self.sources.items():
            try:
                signals[source] = analyzer.get_sentiment(symbol)
            except Exception as e:
                logger.error(f"Failed to get {source} sentiment: {e}")
                signals[source] = {'signal': 0, 'confidence': 0}
        
        # Adaptive weighting based on confidence
        weighted_signal = 0
        total_weight = 0
        
        for source, signal in signals.items():
            if signal['confidence'] > self.confidence_threshold:
                weight = self.weights[source] * signal['confidence']
                weighted_signal += signal['signal'] * weight
                total_weight += weight
                
        if total_weight > 0:
            final_signal = weighted_signal / total_weight
        else:
            final_signal = 0
            
        # Special rules for political signals
        if signals.get('unusual_whales', {}).get('congress_buying', 0) > 0.8:
            final_signal = max(final_signal, 0.7)  # Strong buy signal
            
        return {
            'signal': final_signal,
            'components': signals,
            'confidence': total_weight / sum(self.weights.values())
        }
```

## 🔁 Event-Driven Architecture

### Real-Time Event Processing
```python
# Event Bus Configuration
event_bus = EventBus(
    handlers={
        'price_update': PriceUpdateHandler(),
        'sentiment_spike': SentimentSpikeHandler(),
        'pattern_detected': PatternDetectionHandler(),
        'risk_breach': RiskBreachHandler(),
        'order_filled': OrderFilledHandler(),
        'political_trade': PoliticalTradeHandler()  # New!
    }
)

# Scheduled Tasks
scheduler = AdvancedScheduler(
    tasks=[
        CronTask('0 * * * *', 'rebalance_portfolio'),      # Hourly
        CronTask('*/5 * * * *', 'update_predictions'),     # 5 min
        CronTask('*/1 * * * *', 'check_risk_limits'),      # 1 min
        CronTask('*/15 * * * *', 'scrape_unusual_whales'), # 15 min
        CronTask('0 9 * * *', 'daily_model_retrain'),      # Daily
        MarketOpenTask('update_overnight_gaps'),           # Market open
        MarketCloseTask('generate_daily_report')           # Market close
    ]
)
```

## 🌍 Deployment Guide

### Cost Analysis
```
Monthly Costs Breakdown:
- VPS Hosting: €10-40 (depending on specs)
- Alpaca: €0 (free API)
- Reddit API: €0 (free tier)
- Twitter API: €0-100 (optional, v2 API)
- Alpha Vantage: €0 (free tier, 5 calls/min)
- UnusualWhales: €0 (scraping) or €50/mo (API)
- Total: €10-190/month

One-Time Costs:
- Domain (optional): €10/year
- SSL Certificate: €0 (Let's Encrypt)
```

### VPS Deployment Options

#### Option 1: Hetzner Cloud (Recommended - Best Value)
```bash
# Create Hetzner account and get API token
# Install hcloud CLI
brew install hcloud  # macOS
# or
sudo apt install hcloud  # Ubuntu

# Create VPS (CPX31: 4 vCPU, 8GB RAM, €13/month)
hcloud server create \
  --name quantum-trader \
  --type cpx31 \
  --image ubuntu-22.04 \
  --ssh-key ~/.ssh/id_rsa.pub

# Get IP address
hcloud server list

# SSH into server
ssh root@YOUR_SERVER_IP
```

#### Option 2: DigitalOcean
```bash
# Create droplet via CLI (4GB RAM, 2 vCPU, $24/month)
doctl compute droplet create quantum-trader \
  --size s-2vcpu-4gb \
  --image ubuntu-22-04-x64 \
  --region fra1 \
  --ssh-keys YOUR_SSH_KEY_ID
```

#### Option 3: Local Server (Free but requires 24/7 machine)
```bash
# Use existing hardware or old laptop
# Install Ubuntu Server 22.04
# Configure port forwarding for monitoring
```

### Automated Deployment Script
```bash
#!/bin/bash
# Save as deploy.sh and run on fresh Ubuntu 22.04

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3.10 python3-pip nodejs npm git redis-server postgresql nginx

# Clone repository
git clone https://github.com/yourusername/quantumsentiment.git
cd quantumsentiment

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install MCP servers
./scripts/install_mcp_servers.sh

# Setup PostgreSQL
sudo -u postgres psql -c "CREATE DATABASE quantum;"
sudo -u postgres psql -c "CREATE USER quantum WITH PASSWORD 'your_secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE quantum TO quantum;"

# Configure environment
cp .env.example .env
# IMPORTANT: Edit .env with your API keys
nano .env

# Setup systemd service
sudo cp deployment/quantumsentiment.service /etc/systemd/system/
sudo systemctl enable quantumsentiment
sudo systemctl start quantumsentiment

# Setup Nginx reverse proxy (optional, for web dashboard)
sudo cp deployment/nginx.conf /etc/nginx/sites-available/quantum
sudo ln -s /etc/nginx/sites-available/quantum /etc/nginx/sites-enabled/
sudo systemctl restart nginx

# Setup SSL with Let's Encrypt (optional)
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d yourdomain.com

# Setup monitoring
./scripts/setup_monitoring.sh

echo "Deployment complete! Check status with: sudo systemctl status quantumsentiment"
```

### Docker Deployment (Alternative)
```yaml
# docker-compose.yml
version: '3.8'

services:
  quantum-trader:
    build: .
    restart: unless-stopped
    environment:
      - TRADING_MODE=paper
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
    env_file:
      - .env
    depends_on:
      - redis
      - postgres
      
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes:
      - redis-data:/data
      
  postgres:
    image: postgres:15-alpine
    restart: unless-stopped
    environment:
      POSTGRES_DB: quantum
      POSTGRES_USER: quantum
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
      
  grafana:
    image: grafana/grafana:latest
    restart: unless-stopped
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards
      
volumes:
  redis-data:
  postgres-data:
  grafana-data:
```

### Production Checklist
```bash
# 1. Security hardening
./scripts/harden_security.sh

# 2. Setup backups
crontab -e
# Add: 0 2 * * * /home/quantum/backup.sh

# 3. Configure monitoring alerts
python scripts/setup_alerts.py \
  --telegram-token $TELEGRAM_BOT_TOKEN \
  --alert-on "errors,large_losses,system_down"

# 4. Performance tuning
# Edit /etc/sysctl.conf for network optimization
sudo sysctl -p

# 5. Setup log rotation
sudo cp deployment/logrotate.conf /etc/logrotate.d/quantum

# 6. Test failover
./scripts/test_failover.sh
```

### Scaling Considerations
```python
# For multiple strategies or higher frequency
class DistributedQuantum:
    """
    Scale across multiple VPS instances
    """
    def __init__(self):
        self.workers = [
            TradingWorker('strategy_1', 'vps1.example.com'),
            TradingWorker('strategy_2', 'vps2.example.com'),
            TradingWorker('strategy_3', 'vps3.example.com')
        ]
        self.load_balancer = ConsistentHashBalancer()
        
    def distribute_symbols(self, symbols):
        """Distribute symbols across workers for parallel processing"""
        for symbol in symbols:
            worker = self.load_balancer.get_worker(symbol)
            worker.process_symbol(symbol)
```

## 📝 Logging & Performance Analytics

### Multi-Level Logging
```python
# Structured Logging
logger = StructuredLogger(
    handlers=[
        FileHandler('logs/trading.log', level='INFO'),
        DatabaseHandler('postgresql://...', level='DEBUG'),
        CloudwatchHandler('quantum-sentiment', level='ERROR'),
        TelegramHandler(bot_token, level='CRITICAL')
    ]
)

# Performance Metrics Database
CREATE TABLE trades (
    id UUID PRIMARY KEY,
    timestamp TIMESTAMP,
    symbol VARCHAR(10),
    side VARCHAR(4),
    quantity DECIMAL,
    price DECIMAL,
    slippage DECIMAL,
    commission DECIMAL,
    ml_confidence FLOAT,
    features JSONB,
    signals JSONB,
    pnl DECIMAL,
    sentiment_scores JSONB  -- New: detailed sentiment breakdown
);

CREATE TABLE performance_metrics (
    date DATE PRIMARY KEY,
    total_return DECIMAL,
    sharpe_ratio FLOAT,
    sortino_ratio FLOAT,
    max_drawdown FLOAT,
    win_rate FLOAT,
    profit_factor FLOAT,
    avg_win DECIMAL,
    avg_loss DECIMAL,
    trades_count INT,
    ml_accuracy JSONB,
    sentiment_accuracy JSONB  -- New: track sentiment prediction accuracy
);
```

### Real-Time Dashboard
```python
# Dash/Plotly Dashboard
dashboard = Dashboard(
    pages=[
        OverviewPage(metrics=['pnl', 'sharpe', 'drawdown']),
        PositionsPage(show=['current', 'pending', 'history']),
        SignalsPage(models=['lstm', 'sentiment', 'patterns']),
        RiskPage(monitors=['var', 'exposure', 'correlation']),
        MLPage(performance=['accuracy', 'feature_importance']),
        SentimentPage(sources=['reddit', 'twitter', 'unusual_whales'])  # New!
    ],
    update_frequency='realtime'
)
```

## 🧪 Advanced Backtesting Framework

### Vectorized Backtesting Engine
```python
# High-performance backtester
backtester = VectorizedBacktester(
    data_frequency='1min',
    initial_capital=1000,
    commission_model=RealisticCommissions(
        maker_fee=0.001,
        taker_fee=0.002,
        slippage_model='market_impact'
    ),
    features=[
        'walk_forward_optimization',
        'monte_carlo_simulation',
        'parameter_sensitivity',
        'regime_specific_testing',
        'sentiment_impact_analysis'  # New!
    ]
)

# Sentiment-Aware Backtesting
sentiment_backtest = SentimentBacktester(
    historical_reddit_data='data/reddit_wsb_2022_2024.parquet',
    historical_twitter_data='data/twitter_finance_2022_2024.parquet',
    political_trades_data='data/unusual_whales_congress_2022_2024.csv'
)

# Walk-Forward Analysis
walk_forward = WalkForwardAnalysis(
    training_window=90,  # days
    testing_window=30,   # days
    reoptimize_frequency=7,  # days
    optimization_metric='sharpe_ratio'
)

# Monte Carlo Simulations
monte_carlo = MonteCarloSimulator(
    simulations=10000,
    randomize=['price_paths', 'execution_slippage', 'latency', 'sentiment_noise'],
    confidence_intervals=[0.95, 0.99]
)
```

### Strategy Evaluation
```bash
# Comprehensive backtest with sentiment
python scripts/backtest.py \
  --strategy quantum_ensemble \
  --start 2022-01-01 \
  --end 2024-12-31 \
  --initial-capital 1000 \
  --sentiment-sources "reddit,twitter,unusual_whales" \
  --walk-forward \
  --monte-carlo \
  --report-format html

# Compare strategies
python scripts/compare_strategies.py \
  --strategies "ml_ensemble,sentiment_only,technical_only,political_follow" \
  --metrics "sharpe,sortino,calmar,max_dd,win_rate,sentiment_accuracy" \
  --plot
```

## 🆘 Advanced Troubleshooting

### Sentiment Pipeline Diagnostics
```bash
# Check sentiment data quality
python scripts/diagnose_sentiment.py --source reddit --days 7

# Analyze unusual whales scraping
python scripts/test_scraper.py --url https://unusualwhales.com/politics

# Debug Twitter integration
python scripts/test_twitter.py --query "$SPY OR $BTC" --count 100
```

### ML Model Diagnostics
```bash
# Check model performance degradation
python scripts/diagnose_models.py --model lstm_price --window 7d

# Feature importance analysis
python scripts/feature_analysis.py --top-n 20 --plot

# Prediction accuracy by market regime
python scripts/regime_analysis.py --model all
```

### System Performance
```bash
# Profile code bottlenecks
python -m cProfile -o profile.stats src/main.py
python scripts/analyze_profile.py profile.stats

# Memory usage analysis
python scripts/memory_profiler.py --component feature_pipeline

# Latency monitoring
python scripts/latency_monitor.py --threshold-ms 100
```

### Common Issues
```
Q: "UnusualWhales scraping blocked"
A: They use Cloudflare, our scraper handles it but may need rotation
   python scripts/rotate_proxy.py --service unusual_whales

Q: "Model predictions suddenly degraded"
A: Check for data quality issues, market regime change, or feature drift
   python scripts/diagnose_drift.py --days 30

Q: "High slippage on executions"
A: Analyze order book depth and adjust participation rate
   python scripts/analyze_slippage.py --symbol BTC --days 7

Q: "Memory usage growing over time"
A: Check for data leaks in feature pipeline
   python scripts/check_memory_leaks.py
```

## 🛣 Roadmap

### Current Release (v1.0)
- ✅ Multi-strategy ML ensemble
- ✅ Advanced risk management
- ✅ MCP integration layer
- ✅ Vectorized backtesting
- ✅ Multi-source sentiment (Reddit, Twitter, UnusualWhales)

### v2.0 (Q2 2025)
- 🔄 Reinforcement learning market maker
- 🔄 Cross-exchange arbitrage
- 🔄 Options strategies (delta hedging)
- 🔄 Alternative data integration (satellite, web scraping)
- 🔄 Discord/Telegram sentiment integration

### v3.0 (Q3 2025)
- 📋 Transformer architecture for full market attention
- 📋 Multi-agent competition framework
- 📋 Zero-knowledge proof for strategy verification
- 📋 Quantum computing for portfolio optimization
- 📋 On-chain sentiment from blockchain data

### v4.0 (Q4 2025)
- 🚀 Self-improving neural architecture search
- 🚀 Decentralized strategy marketplace
- 🚀 Brain-computer interface for trader intuition capture
- 🚀 AGI integration (when available)

## 💡 Expert Mode Commands

```bash
# One-liner deployment on fresh Ubuntu VPS
curl -sSL https://raw.githubusercontent.com/you/quantumsentiment/main/scripts/deploy.sh | bash

# Docker Compose Alternative
docker-compose -f docker-compose.prod.yml up -d

# Kubernetes Deployment (for scale)
kubectl apply -f k8s/quantum-sentiment.yaml

# Quick sentiment check
python -m quantum.sentiment --ticker GME --sources all
```

## 🤔 Questions for Your Quant Friend

Before unleashing this beast:

1. "Is it concerning that my trading bot has more parameters than a Boeing 747 flight manual?"

2. "When the ML models achieve consciousness and start trading for themselves, do I still get the profits?"

3. "If I'm using the same indicators everyone else uses, but with more decimal places, is that still alpha?"

---

*Remember: This system is simultaneously the most sophisticated €1,000 experiment and the most expensive way to learn that markets are efficiently random. May the odds be ever in your favor.*
```

```markdown
# .env.example

# === BROKER CREDENTIALS (REQUIRED) ===
ALPACA_API_KEY=your_alpaca_key_here
ALPACA_API_SECRET=your_alpaca_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Change to https://api.alpaca.markets for live

# === DATA SOURCES (REQUIRED) ===
REDDIT_CLIENT_ID=your_reddit_app_id
REDDIT_CLIENT_SECRET=your_reddit_app_secret
REDDIT_USER_AGENT=QuantumSentiment/1.0 by /u/yourusername

ALPHA_VANTAGE_API_KEY=your_free_alpha_vantage_key  # For fundamental data

# === OPTIONAL ENHANCED DATA ===
TWITTER_BEARER_TOKEN=  # If you have Twitter API access (leave empty to disable)
NEWSAPI_KEY=  # From newsapi.org for broader news coverage
UNUSUAL_WHALES_API_KEY=  # If you purchase API access (optional, will scrape if empty)

# === SAFETY CONTROLS (REQUIRED - DO NOT SKIP) ===
TRADING_MODE=paper  # MUST be explicitly set to 'live' for real trading
CONFIRM_REAL_MONEY_TRADING=false  # MUST be 'true' for live trading
MAX_POSITION_SIZE_EUR=50  # Absolute max per position
MAX_DAILY_LOSS_EUR=100  # Circuit breaker - stops all trading if hit
MAX_PORTFOLIO_RISK_PERCENT=50  # Never exceed this % of capital invested

# === MODEL CONFIGURATION (OPTIONAL) ===
ML_MODELS_PATH=./models/trained/  # Where to load trained models
FEATURE_CACHE_SIZE_GB=2  # RAM cache for features
ENABLE_GPU=false  # Set true if CUDA available
MODEL_CONFIDENCE_THRESHOLD=0.65  # Min confidence to trade

# === SENTIMENT CONFIGURATION ===
SENTIMENT_SOURCES=reddit,twitter,unusual_whales,news  # Comma separated, remove any you don't want
REDDIT_SUBREDDITS=wallstreetbets,stocks,cryptocurrency,investing  # Subreddits to monitor
TWITTER_ACCOUNTS=whale_alert,jimcramer,zerohedge  # Twitter accounts to follow
UNUSUAL_WHALES_SCRAPE_DELAY=900  # Seconds between scrapes (15 min default)

# === MCP SERVER PORTS (OPTIONAL - DEFAULTS WORK) ===
ALPACA_MCP_PORT=8080
DEXPAPRIKA_MCP_PORT=8081  
WSB_MCP_PORT=8082
ALPHA_VANTAGE_MCP_PORT=8083

# === DATABASE (OPTIONAL) ===
DATABASE_URL=postgresql://quantum:password@localhost/quantum  # Default: SQLite
REDIS_URL=redis://localhost:6379  # For real-time event bus

# === MONITORING & ALERTS (OPTIONAL) ===
TELEGRAM_BOT_TOKEN=  # For critical alerts
TELEGRAM_CHAT_ID=  # Your chat ID
DISCORD_WEBHOOK_URL=  # For Discord alerts
SENTRY_DSN=  # Error tracking
GRAFANA_API_KEY=  # If using Grafana Cloud

# === DEPLOYMENT SETTINGS ===
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
WORKERS=4  # Number of parallel workers
LOG_LEVEL=INFO  # DEBUG for development
ENABLE_PROFILING=false  # Performance profiling

# === ADVANCED SETTINGS (OPTIONAL) ===
BACKTEST_DATA_PATH=./data/historical/
STRATEGY_CONFIG_PATH=./config/strategies.yaml
SCRAPER_USER_AGENT=Mozilla/5.0 (compatible; QuantumBot/1.0)
PROXY_ROTATION_ENABLED=false  # Enable if scraping gets blocked
PROXY_LIST_PATH=./config/proxies.txt
```