#!/usr/bin/env python3
"""
Example usage of the algorithmic trading bot.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from strategies import MovingAverageStrategy
from config import config


def fetch_sample_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Fetch sample data for demonstration."""
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period)
    return data


def run_example():
    """Run a simple example of the trading strategy."""
    print("Algorithmic Trading Bot - Example")
    print("=" * 40)
    
    # Initialize strategy
    strategy = MovingAverageStrategy(short_window=10, long_window=30)
    print(f"Strategy: {strategy.name}")
    print(f"Parameters: {strategy.params}")
    
    # Fetch sample data
    symbol = "AAPL"
    print(f"\nFetching data for {symbol}...")
    data = fetch_sample_data(symbol, period="6mo")
    
    if data.empty:
        print("No data received. Please check your internet connection.")
        return
    
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
    
    # Generate signals
    print("\nGenerating trading signals...")
    signals_df = strategy.generate_signals(data)
    
    # Show recent signals
    recent_signals = signals_df.tail(10)[['close', 'short_ma', 'long_ma', 'signal']]
    print("\nRecent signals:")
    print(recent_signals)
    
    # Calculate position sizes
    print("\nCalculating position sizes...")
    portfolio_value = 10000  # $10,000 portfolio
    latest_price = data['close'].iloc[-1]
    latest_signal = signals_df['signal'].iloc[-1]
    
    position_size = strategy.calculate_position_size(
        latest_signal, latest_price, portfolio_value
    )
    
    print(f"Latest price: ${latest_price:.2f}")
    print(f"Latest signal: {latest_signal}")
    print(f"Position size: {position_size:.2f} shares")
    
    if position_size > 0:
        print("Action: BUY")
    elif position_size < 0:
        print("Action: SELL")
    else:
        print("Action: HOLD")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    run_example() 