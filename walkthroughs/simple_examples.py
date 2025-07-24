#!/usr/bin/env python3
"""
🎯 Simple Trading Model Examples
==============================

Quick, focused examples of each trading model.
Perfect for understanding one concept at a time!

Choose which example to run:
1. LSTM Price Prediction
2. CNN Pattern Recognition  
3. XGBoost Market Regime
4. Risk Calculator
5. Quick Backtest
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def create_sample_data(days=30):
    """Create simple sample trading data"""
    print("📊 Creating sample stock data...")
    
    dates = pd.date_range(start='2024-01-01', periods=days*24, freq='1H')
    
    # Start at $100, add random walk
    prices = [100]
    for _ in range(len(dates)-1):
        change = np.random.normal(0, 0.01)  # 1% volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1))  # Don't go below $1
    
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Simple OHLCV data
        open_price = prices[i-1] if i > 0 else price
        high = max(open_price, price) * (1 + abs(np.random.normal(0, 0.005)))
        low = min(open_price, price) * (1 - abs(np.random.normal(0, 0.005)))
        volume = np.random.randint(100000, 1000000)
        
        data.append({
            'timestamp': date,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(price, 2),
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    print(f"✅ Created {len(df)} hours of data")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    return df

def example_lstm_prediction():
    """Simple LSTM price prediction example"""
    print("\n" + "="*50)
    print("🧠 LSTM PRICE PREDICTION EXAMPLE")
    print("="*50)
    
    print("\nWhat this does:")
    print("• Looks at past price movements")
    print("• Learns patterns in the data")
    print("• Predicts next price change")
    
    try:
        from src.models.lstm.price_lstm import PriceLSTM, PriceLSTMConfig
        
        # Create data
        data = create_sample_data(days=10)
        
        # Simple config
        config = PriceLSTMConfig(
            sequence_length=12,  # Look back 12 hours
            forecast_horizon=1,  # Predict 1 hour ahead
            epochs=3,            # Quick training
            batch_size=8,
            save_path=Path("simple_demo/lstm")
        )
        
        print(f"\n⚙️ Model setup:")
        print(f"• Looks back: {config.sequence_length} hours")
        print(f"• Predicts: {config.forecast_horizon} hour ahead")
        
        # Create and train model
        model = PriceLSTM(config)
        
        # Prepare data
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        
        features = train_data[['open', 'high', 'low', 'close', 'volume']]
        # Target: next hour's price change
        train_data['next_return'] = train_data['close'].pct_change().shift(-1)
        targets = train_data['next_return'].fillna(0)
        
        print(f"\n🎓 Training on {len(train_data)} samples...")
        history = model.train(features, targets)
        print("✅ Training complete!")
        
        # Make predictions
        test_data = data[train_size:]
        if len(test_data) > 12:  # Need enough data for sequence
            test_features = test_data[['open', 'high', 'low', 'close', 'volume']]
            predictions = model.predict(test_features)
            
            print(f"\n🔮 Predictions for next {len(predictions)} hours:")
            for i in range(min(5, len(predictions))):
                current_price = test_data.iloc[i]['close']
                predicted_change = predictions[i] * 100
                predicted_price = current_price * (1 + predictions[i])
                
                direction = "📈 UP" if predictions[i] > 0 else "📉 DOWN"
                print(f"Hour {i+1}: ${current_price:.2f} → ${predicted_price:.2f} ({predicted_change:+.2f}%) {direction}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("This might happen due to data size or dependencies")

def example_cnn_patterns():
    """Simple CNN pattern recognition example"""
    print("\n" + "="*50)
    print("👁️ CNN PATTERN RECOGNITION EXAMPLE")
    print("="*50)
    
    print("\nWhat this does:")
    print("• Converts price data into chart images")
    print("• Recognizes visual patterns")
    print("• Identifies bullish/bearish formations")
    
    try:
        from src.models.cnn.chart_pattern_cnn import ChartPatternCNN, ChartPatternConfig
        
        # Create data
        data = create_sample_data(days=15)
        
        # Simple config
        config = ChartPatternConfig(
            chart_height=16,     # Small chart for demo
            chart_width=32,
            epochs=2,            # Quick training
            batch_size=4,
            save_path=Path("simple_demo/cnn")
        )
        
        print(f"\n⚙️ Model setup:")
        print(f"• Chart size: {config.chart_width}x{config.chart_height} pixels")
        print(f"• Can recognize: {len(config.pattern_classes)} patterns")
        
        # Create model
        model = ChartPatternCNN(config)
        
        # Create simple pattern labels
        train_data = data[:int(len(data) * 0.8)].copy()
        
        # Label based on price movement
        price_change = train_data['close'].pct_change(5)  # 5-period change
        train_data['pattern'] = 'sideways'  # default
        
        # Simple pattern rules
        train_data.loc[price_change > 0.02, 'pattern'] = 'bull_flag'  # Up 2%+
        train_data.loc[price_change < -0.02, 'pattern'] = 'bear_flag'  # Down 2%+
        
        patterns = train_data['pattern']
        print(f"\n📊 Pattern distribution:")
        for pattern, count in patterns.value_counts().items():
            print(f"• {pattern.replace('_', ' ').title()}: {count}")
        
        print(f"\n🎓 Training pattern recognition...")
        history = model.train(train_data, patterns)
        print("✅ Training complete!")
        
        # Test pattern recognition
        test_data = data[int(len(data) * 0.8):]
        if len(test_data) > 32:  # Need enough data
            predictions = model.predict(test_data)
            
            if len(predictions) > 0:
                print(f"\n🔍 Detected patterns:")
                unique_patterns = np.unique(predictions)
                for pattern in unique_patterns:
                    count = np.sum(predictions == pattern)
                    print(f"• {pattern.replace('_', ' ').title()}: {count} times")
                
                # Trading advice
                if 'bull_flag' in predictions:
                    print("\n💡 Trading signal: BULLISH patterns detected!")
                elif 'bear_flag' in predictions:
                    print("\n💡 Trading signal: BEARISH patterns detected!")
                else:
                    print("\n💡 Trading signal: No clear patterns - stay cautious")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Pattern recognition needs more data or different parameters")

def example_xgboost_regime():
    """Simple XGBoost market regime example"""
    print("\n" + "="*50)
    print("🏛️ XGBOOST MARKET REGIME EXAMPLE")
    print("="*50)
    
    print("\nWhat this does:")
    print("• Calculates technical indicators")
    print("• Determines market 'mood'")
    print("• Suggests trading approach")
    
    try:
        from src.models.xgboost.market_regime_xgboost import MarketRegimeXGBoost, MarketRegimeConfig
        
        # Create data
        data = create_sample_data(days=20)
        
        # Simple config
        config = MarketRegimeConfig(
            n_estimators=20,  # Small forest for demo
            max_depth=3,
            save_path=Path("simple_demo/xgboost")
        )
        
        print(f"\n⚙️ Model setup:")
        print(f"• Decision trees: {config.n_estimators}")
        print(f"• Market regimes: {len(config.regime_classes)}")
        
        print(f"\n🏷️ Market regimes it detects:")
        regime_descriptions = {
            'strong_bull': '🚀 Prices rising fast',
            'bull': '📈 Generally rising',
            'sideways': '↔️ Range-bound',
            'bear': '📉 Generally falling',
            'strong_bear': '💥 Prices falling fast',
            'high_volatility': '🌪️ Very unstable'
        }
        
        for regime in config.regime_classes:
            desc = regime_descriptions.get(regime, 'Market condition')
            print(f"• {regime.replace('_', ' ').title()}: {desc}")
        
        # Create and train model
        model = MarketRegimeXGBoost(config)
        
        train_data = data[:int(len(data) * 0.8)]
        print(f"\n🎓 Training on {len(train_data)} samples...")
        
        # Model will auto-detect regimes from price patterns
        history = model.train(train_data, None)
        print("✅ Training complete!")
        
        # Show important features
        if hasattr(model, 'feature_importance'):
            print(f"\n🧠 Most important indicators:")
            important_features = sorted(model.feature_importance.items(), 
                                      key=lambda x: x[1], reverse=True)[:3]
            for feature, importance in important_features:
                print(f"• {feature}: {importance:.3f}")
        
        # Detect current regime
        test_data = data[int(len(data) * 0.8):]
        if len(test_data) > 50:  # Need enough data for indicators
            predictions = model.predict(test_data)
            
            if len(predictions) > 0:
                # Most common regime
                unique_regimes, counts = np.unique(predictions, return_counts=True)
                most_common_idx = np.argmax(counts)
                current_regime = unique_regimes[most_common_idx]
                
                print(f"\n🎯 Current market regime: {current_regime.replace('_', ' ').title()}")
                
                # Trading advice based on regime
                trading_advice = {
                    'strong_bull': "💪 Strong uptrend - consider buying dips",
                    'bull': "👍 Uptrend - look for buying opportunities",
                    'sideways': "⚖️ Range trading - buy low, sell high",
                    'bear': "👎 Downtrend - consider selling or shorting",
                    'strong_bear': "💀 Strong downtrend - avoid buying",
                    'high_volatility': "⚠️ High risk - reduce position sizes"
                }
                
                advice = trading_advice.get(current_regime, "Stay cautious")
                print(f"💡 Trading advice: {advice}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Market regime detection needs sufficient data")

def example_risk_calculator():
    """Simple risk management calculator"""
    print("\n" + "="*50)
    print("🛡️ RISK MANAGEMENT CALCULATOR")
    print("="*50)
    
    print("\nWhat this does:")
    print("• Calculates safe position sizes")
    print("• Shows maximum risk per trade")
    print("• Helps protect your capital")
    
    # Get user input (or use defaults)
    account_size = 10000  # $10,000
    risk_percent = 2      # 2% risk per trade
    entry_price = 100     # $100 stock price
    stop_loss_percent = 5 # 5% stop loss
    
    print(f"\n💰 Account Setup:")
    print(f"• Account size: ${account_size:,}")
    print(f"• Risk per trade: {risk_percent}%")
    print(f"• Entry price: ${entry_price}")
    print(f"• Stop loss: {stop_loss_percent}% below entry")
    
    # Calculate position size
    max_risk_amount = account_size * (risk_percent / 100)
    stop_loss_price = entry_price * (1 - stop_loss_percent / 100)
    risk_per_share = entry_price - stop_loss_price
    max_shares = int(max_risk_amount / risk_per_share)
    position_value = max_shares * entry_price
    
    print(f"\n📊 Risk Calculation:")
    print(f"• Max risk amount: ${max_risk_amount:,.0f}")
    print(f"• Stop loss price: ${stop_loss_price:.2f}")
    print(f"• Risk per share: ${risk_per_share:.2f}")
    print(f"• Max shares to buy: {max_shares:,}")
    print(f"• Position value: ${position_value:,}")
    
    # Show scenarios
    print(f"\n🎭 What-if scenarios:")
    
    # Win scenario
    target_price = entry_price * 1.10  # 10% gain target
    profit_if_win = max_shares * (target_price - entry_price)
    
    print(f"✅ If price hits ${target_price:.2f} (+10%):")
    print(f"   Profit: ${profit_if_win:,.0f} ({profit_if_win/account_size*100:.1f}% of account)")
    
    # Loss scenario
    loss_if_stopped = max_shares * risk_per_share
    print(f"❌ If stopped out at ${stop_loss_price:.2f}:")
    print(f"   Loss: ${loss_if_stopped:,.0f} ({loss_if_stopped/account_size*100:.1f}% of account)")
    
    # Risk-reward ratio
    risk_reward_ratio = (target_price - entry_price) / (entry_price - stop_loss_price)
    print(f"⚖️ Risk-reward ratio: 1:{risk_reward_ratio:.1f}")
    
    print(f"\n💡 Risk Management Tips:")
    print(f"• Never risk more than 2% per trade")
    print(f"• Always use stop losses")
    print(f"• Look for 2:1 or better risk-reward ratios")
    print(f"• Size positions based on risk, not gut feeling")

def example_quick_backtest():
    """Simple backtesting example"""
    print("\n" + "="*50)
    print("📈 QUICK BACKTEST EXAMPLE")
    print("="*50)
    
    print("\nWhat this does:")
    print("• Tests a simple trading strategy")
    print("• Shows how it would have performed")
    print("• Calculates key performance metrics")
    
    # Create data
    data = create_sample_data(days=30)
    
    print(f"\n📊 Testing Strategy: 'Buy Low, Sell High'")
    print(f"• Buy when price drops 3% from recent high")
    print(f"• Sell when price rises 5% from purchase")
    print(f"• Stop loss at 2% below purchase price")
    
    # Simple strategy backtest
    starting_cash = 10000
    cash = starting_cash
    shares = 0
    trade_history = []
    
    # Calculate rolling high
    data['high_20'] = data['close'].rolling(20).max()
    data['drop_from_high'] = (data['close'] - data['high_20']) / data['high_20']
    
    for i in range(20, len(data)):  # Start after rolling window
        current_price = data.iloc[i]['close']
        drop_pct = data.iloc[i]['drop_from_high']
        
        # Buy signal: dropped 3% from 20-period high
        if shares == 0 and drop_pct <= -0.03 and cash > current_price:
            # Buy as many shares as we can afford
            shares_to_buy = int(cash * 0.95 / current_price)  # Use 95% of cash
            if shares_to_buy > 0:
                shares = shares_to_buy
                cost = shares * current_price
                cash -= cost
                
                trade_history.append({
                    'action': 'BUY',
                    'price': current_price,
                    'shares': shares,
                    'value': cost,
                    'date': data.iloc[i]['timestamp']
                })
        
        # Sell signals
        elif shares > 0:
            last_buy_price = [t['price'] for t in trade_history if t['action'] == 'BUY'][-1]
            
            # Take profit: 5% gain
            if current_price >= last_buy_price * 1.05:
                proceeds = shares * current_price
                cash += proceeds
                
                trade_history.append({
                    'action': 'SELL',
                    'price': current_price,
                    'shares': shares,
                    'value': proceeds,
                    'date': data.iloc[i]['timestamp']
                })
                shares = 0
            
            # Stop loss: 2% loss
            elif current_price <= last_buy_price * 0.98:
                proceeds = shares * current_price
                cash += proceeds
                
                trade_history.append({
                    'action': 'STOP',
                    'price': current_price,
                    'shares': shares,
                    'value': proceeds,
                    'date': data.iloc[i]['timestamp']
                })
                shares = 0
    
    # Final portfolio value
    final_price = data.iloc[-1]['close']
    final_value = cash + (shares * final_price)
    total_return = (final_value - starting_cash) / starting_cash * 100
    
    print(f"\n📊 Backtest Results:")
    print(f"• Starting value: ${starting_cash:,}")
    print(f"• Final value: ${final_value:,.0f}")
    print(f"• Total return: {total_return:+.1f}%")
    print(f"• Number of trades: {len(trade_history)}")
    
    # Show trade history
    if trade_history:
        print(f"\n📋 Trade History:")
        for i, trade in enumerate(trade_history[-6:]):  # Show last 6 trades
            action_emoji = {'BUY': '🟢', 'SELL': '✅', 'STOP': '🛑'}
            emoji = action_emoji.get(trade['action'], '⚪')
            print(f"{emoji} {trade['action']}: {trade['shares']} shares at ${trade['price']:.2f}")
    
    # Performance metrics
    buy_trades = [t for t in trade_history if t['action'] == 'BUY']
    sell_trades = [t for t in trade_history if t['action'] in ['SELL', 'STOP']]
    
    if len(buy_trades) > 0 and len(sell_trades) > 0:
        winning_trades = 0
        total_profit = 0
        
        for i, sell in enumerate(sell_trades):
            if i < len(buy_trades):
                buy = buy_trades[i]
                profit = (sell['price'] - buy['price']) / buy['price'] * 100
                total_profit += profit
                if profit > 0:
                    winning_trades += 1
        
        win_rate = winning_trades / len(sell_trades) * 100
        avg_return = total_profit / len(sell_trades)
        
        print(f"\n📈 Strategy Performance:")
        print(f"• Win rate: {win_rate:.1f}%")
        print(f"• Average return per trade: {avg_return:+.1f}%")
        
        if total_return > 0:
            print("✅ Strategy was profitable!")
        else:
            print("❌ Strategy lost money - needs improvement")

def main():
    """Run the simple examples"""
    print("🎯 Simple Trading Model Examples")
    print("=" * 40)
    print("Choose an example to run:")
    print("1. LSTM Price Prediction")
    print("2. CNN Pattern Recognition")
    print("3. XGBoost Market Regime")
    print("4. Risk Management Calculator")
    print("5. Quick Backtesting")
    print("6. Run All Examples")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    # Create output directory
    os.makedirs("simple_demo", exist_ok=True)
    
    if choice == '1':
        example_lstm_prediction()
    elif choice == '2':
        example_cnn_patterns()
    elif choice == '3':
        example_xgboost_regime()
    elif choice == '4':
        example_risk_calculator()
    elif choice == '5':
        example_quick_backtest()
    elif choice == '6':
        print("🚀 Running all examples...")
        example_lstm_prediction()
        example_cnn_patterns() 
        example_xgboost_regime()
        example_risk_calculator()
        example_quick_backtest()
    else:
        print("Invalid choice. Running all examples...")
        example_lstm_prediction()
        example_cnn_patterns()
        example_xgboost_regime()
        example_risk_calculator()
        example_quick_backtest()
    
    print(f"\n🎉 Examples complete!")
    print(f"⚠️  Remember: This is educational only - not investment advice!")

if __name__ == "__main__":
    main()