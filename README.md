# QuantumSentiment Algorithmic Trading Bot 🚀

A state-of-the-art algorithmic trading system that combines:
- 🧠 **AI-Powered Predictions**: CNN, LSTM, XGBoost, and Transformer models
- 💬 **Sentiment Analysis**: Reddit and news sentiment fusion
- 📊 **Smart Portfolio Management**: Regime-aware optimization
- ⚡ **Advanced Execution**: TWAP, VWAP, and Iceberg strategies
- 🛡️ **Comprehensive Risk Management**: Multi-layer protection
- 📈 **Full Broker Integration**: Complete Alpaca API integration


## 📚 Documentation

Comprehensive guides are available in the `docs/guides/` directory:

1. **[Quick Start Guide](docs/guides/quick_start.md)** - Get trading in 5 minutes
2. **[Getting Started](docs/guides/getting_started.md)** - Detailed setup instructions
3. **[System Architecture](docs/guides/system_architecture.md)** - How everything works together
4. **[Trading Strategies](docs/guides/trading_strategies.md)** - Strategy details and customization
5. **[Risk Management](docs/guides/risk_management.md)** - Risk controls and safety features

## 🎯 System Capabilities

### Intelligent Signal Generation
- **Multi-Model Ensemble**: Combines predictions from 4 specialized AI models
- **Sentiment Integration**: Analyzes Reddit posts and news articles
- **108+ Features**: Technical indicators, microstructure, and sentiment metrics
- **Confidence Scoring**: Only trades high-confidence signals

### Smart Execution
- **Slippage Prediction**: XGBoost model predicts and minimizes slippage
- **Market Impact Models**: Kyle's Lambda and Almgren-Chriss optimization
- **Adaptive Strategies**: TWAP, VWAP, and Iceberg based on conditions
- **Multi-Venue Support**: Ready for expansion beyond Alpaca

### Risk Management
- **Position Controls**: Stop loss, take profit, position sizing
- **Portfolio Limits**: Drawdown protection, concentration limits
- **Real-time Monitoring**: Continuous risk assessment
- **Emergency Stops**: Automatic circuit breakers

### Performance Tracking
- **Real-time P&L**: Track unrealized and realized gains
- **Risk Metrics**: Sharpe ratio, VaR, maximum drawdown
- **Trade Analytics**: Win rate, risk/reward, execution quality
- **Account Monitoring**: Margin, buying power, PDT compliance

## 🔧 Configuration Options

### Trading Modes
- **full_auto**: Fully autonomous trading
- **semi_auto**: Generates signals, requires approval
- **paper**: Paper trading with Alpaca
- **backtest**: Historical strategy testing

### Risk Profiles
Edit `config/config.yaml` to adjust risk:
```yaml
# Conservative
risk:
  max_position_size: 0.05  # 5% max
  stop_loss_pct: 0.01     # 1% stop
  max_drawdown: 0.05      # 5% max DD

# Moderate (default)
risk:
  max_position_size: 0.10  # 10% max
  stop_loss_pct: 0.02     # 2% stop
  max_drawdown: 0.10      # 10% max DD

# Aggressive
risk:
  max_position_size: 0.15  # 15% max
  stop_loss_pct: 0.03     # 3% stop
  max_drawdown: 0.15      # 15% max DD
```

## 📊 Monitoring Your Bot

### Check Status
```bash
python scripts/check_status.py
```

### View Real-time Logs
```bash
tail -f logs/trading.log
```

### Performance Dashboard
```bash
python scripts/run_dashboard.py
# Open http://localhost:8050
```

### Daily Summary
```bash
python scripts/daily_summary.py
```

## 🛠️ Useful Scripts

- `scripts/verify_setup.py` - Verify your setup is correct
- `scripts/check_status.py` - Check current system status
- `scripts/emergency_flatten.py` - Emergency position closure
- `scripts/show_positions.py` - Display current positions
- `scripts/generate_report.py` - Generate performance reports

## 🚀 What's Next?

### 1. Paper Trade for 1-2 Weeks
- Monitor daily performance
- Understand the bot's behavior
- Fine-tune parameters

### 2. Analyze Results
- Review win rate and risk/reward
- Identify best-performing strategies
- Optimize position sizing

### 3. Graduate to Live Trading
- Start with minimal capital
- Gradually increase as confidence grows
- Always monitor actively

### 4. Continuous Improvement
- Add new data sources
- Train on recent data
- Implement new strategies
- Join the community

## ⚠️ Important Disclaimers

1. **This is not financial advice** - Trading involves substantial risk
2. **Start with paper trading** - Always test thoroughly before using real money
3. **Monitor regularly** - Automated doesn't mean unattended
4. **Understand the risks** - You can lose money, even with sophisticated algorithms
5. **Past performance doesn't guarantee future results**

## 🤝 Community and Support

- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Share strategies and results
- **Documentation**: Continuously updated guides
- **Updates**: Regular improvements and bug fixes

## 🎊 Congratulations!

You now have a professional-grade algorithmic trading system at your disposal. The combination of:
- Advanced AI models
- Real-time sentiment analysis  
- Sophisticated risk management
- Smart order execution

...gives you a significant edge in the markets.

**Remember**: Start small, learn continuously, and always prioritize risk management.

Happy Trading! May your algorithms be profitable and your drawdowns be minimal! 🚀📈

---

*Built with passion for quantitative trading and AI*