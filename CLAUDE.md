# CLAUDE.md - Essential Memory

## Project: QuantumSentiment AI Trading System
**Budget**: €1,000 | **Mode**: Paper first, then live | **Broker**: Alpaca

## Architecture Evolution
1. **Phase 1-5**: Direct APIs → ML → Risk → Execution
2. **Phase 6**: + LLM intelligence layer
3. **Phase 7**: MCP + LLM architecture

## Critical Safety
- Default: Paper trading only
- Live requires: CONFIRM_REAL_MONEY_TRADING=true
- Max: €50/position, €100/day loss, 20% drawdown

## Modularity Rules
- `src/` for core modules (data, models, risk, execution)
- Each component: single responsibility, clear interfaces
- No monolithic files - break into logical modules
- Type hints everywhere, comprehensive logging

## Key Reminders
- **Always** update TODO.md as you implement features and note changes in NOTES.md 
- **Always** reference README.md for complete requirements
- **Always** download packages with source .venv/bin/activate
- Start with direct APIs, add LLM layer later, then MCP
- Paper trading must work perfectly before live trading
- Risk management is more important than returns

## Dependencies
- **APIs**: alpaca-trade-api, praw, tweepy, alpha-vantage
- **ML**: torch, transformers, xgboost, scikit-learn  
- **Risk**: quantlib, scipy, pandas
- **Data**: postgresql, redis, sqlalchemy