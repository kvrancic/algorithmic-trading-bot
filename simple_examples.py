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

# Add the parent directory to the path so we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def create_sample_data(days=30):
    """Create simple sample trading data"""
    print("📊 Creating sample stock data...")
    
    dates = pd.date_range(start='2024-01-01', periods=days*24, freq='1H')
    
    # Start at $100, add more realistic price movements
    prices = [100]
    for i in range(len(dates)-1):
        # Add trend and cycle components for more realistic data
        trend = 0.0001 * np.sin(i / 100)  # Long-term trend
        cycle = 0.005 * np.sin(i / 24)    # Daily cycle
        noise = np.random.normal(0, 0.015)  # 1.5% volatility
        
        change = trend + cycle + noise
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
        
        # Create extensive data for real trading model
        data = create_sample_data(days=365)  # Full year of hourly data
        
        # Production-ready config for real trading
        config = PriceLSTMConfig(
            sequence_length=48,   # Look back 48 hours (2 full days)
            forecast_horizon=1,   # Predict 1 hour ahead
            lstm_hidden_size=128, # Larger network for complex patterns
            lstm_layers=3,        # Three layers for deeper learning
            epochs=100,           # More extensive training
            batch_size=32,        # Smaller batch for better gradients
            learning_rate=0.0005, # More conservative learning rate
            dropout_rate=0.2,     # Less dropout for more learning
            scaling_method="none",# NO SCALING - predict returns directly
            use_scheduler=True,   # Use learning rate scheduler
            scheduler_type="cosine",
            early_stopping_patience=15,  # More patience for convergence
            use_attention=True,   # Enable attention mechanism
            attention_heads=8,    # More attention heads
            save_path=Path("simple_demo/lstm")
        )
        
        print(f"\n⚙️ Production Model Setup:")
        print(f"• Training data: {len(data)} hours ({len(data)/24:.0f} days)")
        print(f"• Looks back: {config.sequence_length} hours")
        print(f"• Predicts: {config.forecast_horizon} hour ahead")
        print(f"• Network size: {config.lstm_hidden_size} neurons x {config.lstm_layers} layers")
        print(f"• Max epochs: {config.epochs} (with early stopping)")
        print(f"• Batch size: {config.batch_size}")
        
        # Create and train model
        model = PriceLSTM(config)
        
        # Split data for proper time series validation (chronological)
        train_size = int(len(data) * 0.7)  # 70% train
        val_size = int(len(data) * 0.15)   # 15% validation
        test_size = len(data) - train_size - val_size  # 15% test
        
        train_data = data[:train_size].copy()
        val_data = data[train_size:train_size+val_size].copy()
        test_data = data[train_size+val_size:].copy()
        
        print(f"• Train: {len(train_data)} hours ({len(train_data)/24:.0f} days)")
        print(f"• Validation: {len(val_data)} hours ({len(val_data)/24:.0f} days)")
        print(f"• Test: {len(test_data)} hours ({len(test_data)/24:.0f} days)")
        
        # Enhanced feature engineering with technical indicators
        features_normalized = pd.DataFrame()
        
        # Price features (percentage changes)
        features_normalized['open_pct'] = train_data['open'].pct_change().fillna(0)
        features_normalized['high_pct'] = train_data['high'].pct_change().fillna(0)
        features_normalized['low_pct'] = train_data['low'].pct_change().fillna(0)
        features_normalized['close_pct'] = train_data['close'].pct_change().fillna(0)
        
        # Volume features
        features_normalized['volume_norm'] = (train_data['volume'] - train_data['volume'].min()) / (train_data['volume'].max() - train_data['volume'].min())
        features_normalized['volume_pct'] = train_data['volume'].pct_change().fillna(0)
        
        # Technical indicators
        # Moving averages
        ma_5 = train_data['close'].rolling(5).mean()
        ma_20 = train_data['close'].rolling(20).mean()
        features_normalized['ma_5_pct'] = (train_data['close'] - ma_5) / ma_5
        features_normalized['ma_20_pct'] = (train_data['close'] - ma_20) / ma_20
        features_normalized['ma_crossover'] = (ma_5 - ma_20) / ma_20
        
        # RSI-like momentum indicator
        close_diff = train_data['close'].diff()
        gain = close_diff.where(close_diff > 0, 0).rolling(14).mean()
        loss = (-close_diff.where(close_diff < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)  # Avoid division by zero
        features_normalized['rsi'] = (rs / (1 + rs) - 0.5) * 2  # Normalize to -1 to 1
        
        # Price position within range
        high_14 = train_data['high'].rolling(14).max()
        low_14 = train_data['low'].rolling(14).min()
        features_normalized['price_position'] = (train_data['close'] - low_14) / (high_14 - low_14 + 1e-8)
        
        # Volatility
        features_normalized['volatility'] = train_data['close'].pct_change().rolling(10).std().fillna(0)
        
        # Fill any remaining NaN values
        features_normalized = features_normalized.fillna(0)
        
        # Target: next hour's returns (clean)
        future_returns = train_data['close'].pct_change().shift(-1).fillna(0)
        
        # Remove extreme outliers more aggressively
        targets = np.clip(future_returns, -0.05, 0.05)  # Clip to ±5%
        
        features = features_normalized
        
        print(f"• Target range: {targets.min():.4f} to {targets.max():.4f}")
        print(f"• Target mean: {targets.mean():.5f}, std: {targets.std():.4f}")
        
        # Ensure we have enough data after removing NaNs
        features = features[:-1]  # Remove last row to match targets
        targets = targets[:-1]    # Remove last row which is NaN
        
        print(f"\n🎓 Training production model...")
        print(f"Training on {len(features)} samples...")
        
        # Prepare validation data with same feature engineering
        val_features_norm = pd.DataFrame()
        
        # Price features
        val_features_norm['open_pct'] = val_data['open'].pct_change().fillna(0)
        val_features_norm['high_pct'] = val_data['high'].pct_change().fillna(0)
        val_features_norm['low_pct'] = val_data['low'].pct_change().fillna(0)
        val_features_norm['close_pct'] = val_data['close'].pct_change().fillna(0)
        
        # Volume features (use train stats for normalization)
        vol_min, vol_max = train_data['volume'].min(), train_data['volume'].max()
        val_features_norm['volume_norm'] = (val_data['volume'] - vol_min) / (vol_max - vol_min)
        val_features_norm['volume_pct'] = val_data['volume'].pct_change().fillna(0)
        
        # Technical indicators
        val_ma_5 = val_data['close'].rolling(5).mean()
        val_ma_20 = val_data['close'].rolling(20).mean()
        val_features_norm['ma_5_pct'] = (val_data['close'] - val_ma_5) / val_ma_5
        val_features_norm['ma_20_pct'] = (val_data['close'] - val_ma_20) / val_ma_20
        val_features_norm['ma_crossover'] = (val_ma_5 - val_ma_20) / val_ma_20
        
        # RSI
        val_close_diff = val_data['close'].diff()
        val_gain = val_close_diff.where(val_close_diff > 0, 0).rolling(14).mean()
        val_loss = (-val_close_diff.where(val_close_diff < 0, 0)).rolling(14).mean()
        val_rs = val_gain / (val_loss + 1e-8)
        val_features_norm['rsi'] = (val_rs / (1 + val_rs) - 0.5) * 2
        
        # Price position
        val_high_14 = val_data['high'].rolling(14).max()
        val_low_14 = val_data['low'].rolling(14).min()
        val_features_norm['price_position'] = (val_data['close'] - val_low_14) / (val_high_14 - val_low_14 + 1e-8)
        
        # Volatility
        val_features_norm['volatility'] = val_data['close'].pct_change().rolling(10).std().fillna(0)
        val_features_norm = val_features_norm.fillna(0)
        
        val_targets = np.clip(val_data['close'].pct_change().shift(-1).fillna(0), -0.05, 0.05)
        val_features_norm = val_features_norm[:-1]
        val_targets = val_targets[:-1]
        
        # Train with validation data
        history = model.train(features, targets, validation_data=(val_features_norm, val_targets))
        print("✅ Training complete!")
        print(f"• Epochs trained: {len(history['train_loss'])}")
        print(f"• Final training loss: {history['train_loss'][-1]:.6f}")
        print(f"• Final validation loss: {history['val_loss'][-1]:.6f}")
        print(f"• Best validation loss: {min(history['val_loss']):.6f}")
        
        # COMPREHENSIVE TESTING ON HELD-OUT DATA
        print(f"\n🧪 TESTING MODEL ON UNSEEN DATA ({len(test_data)} hours)")
        
        # Apply same feature engineering to test data
        test_features_normalized = pd.DataFrame()
        
        # Price features
        test_features_normalized['open_pct'] = test_data['open'].pct_change().fillna(0)
        test_features_normalized['high_pct'] = test_data['high'].pct_change().fillna(0)
        test_features_normalized['low_pct'] = test_data['low'].pct_change().fillna(0)
        test_features_normalized['close_pct'] = test_data['close'].pct_change().fillna(0)
        
        # Volume features (use train stats)
        vol_min, vol_max = train_data['volume'].min(), train_data['volume'].max()
        test_features_normalized['volume_norm'] = (test_data['volume'] - vol_min) / (vol_max - vol_min)
        test_features_normalized['volume_pct'] = test_data['volume'].pct_change().fillna(0)
        
        # Technical indicators
        test_ma_5 = test_data['close'].rolling(5).mean()
        test_ma_20 = test_data['close'].rolling(20).mean()
        test_features_normalized['ma_5_pct'] = (test_data['close'] - test_ma_5) / test_ma_5
        test_features_normalized['ma_20_pct'] = (test_data['close'] - test_ma_20) / test_ma_20
        test_features_normalized['ma_crossover'] = (test_ma_5 - test_ma_20) / test_ma_20
        
        # RSI
        test_close_diff = test_data['close'].diff()
        test_gain = test_close_diff.where(test_close_diff > 0, 0).rolling(14).mean()
        test_loss = (-test_close_diff.where(test_close_diff < 0, 0)).rolling(14).mean()
        test_rs = test_gain / (test_loss + 1e-8)
        test_features_normalized['rsi'] = (test_rs / (1 + test_rs) - 0.5) * 2
        
        # Price position
        test_high_14 = test_data['high'].rolling(14).max()
        test_low_14 = test_data['low'].rolling(14).min()
        test_features_normalized['price_position'] = (test_data['close'] - test_low_14) / (test_high_14 - test_low_14 + 1e-8)
        
        # Volatility
        test_features_normalized['volatility'] = test_data['close'].pct_change().rolling(10).std().fillna(0)
        test_features_normalized = test_features_normalized.fillna(0)
        
        predictions = model.predict(test_features_normalized)
        
        # Calculate actual returns for comparison
        actual_returns = test_data['close'].pct_change().shift(-1).fillna(0)
        actual_returns = actual_returns[:len(predictions)]  # Match prediction length
        
        print(f"\n📊 PRODUCTION MODEL PERFORMANCE:")
        print(f"• Total predictions made: {len(predictions)}")
        
        # Prediction quality metrics
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        actual_mean = np.mean(actual_returns)
        actual_std = np.std(actual_returns)
        
        print(f"• Predicted avg return: {pred_mean*100:+.4f}% per hour")
        print(f"• Actual avg return: {actual_mean*100:+.4f}% per hour")
        print(f"• Prediction volatility: {pred_std*100:.4f}%")
        print(f"• Actual volatility: {actual_std*100:.4f}%")
        
        # Ensure predictions and actual returns are 1D arrays
        if hasattr(predictions, 'flatten'):
            predictions_flat = predictions.flatten()
        else:
            predictions_flat = np.array(predictions).flatten()
            
        if hasattr(actual_returns, 'values'):
            actual_returns_flat = actual_returns.values
            if hasattr(actual_returns_flat, 'flatten'):
                actual_returns_flat = actual_returns_flat.flatten()
            else:
                actual_returns_flat = np.array(actual_returns_flat).flatten()
        else:
            if hasattr(actual_returns, 'flatten'):
                actual_returns_flat = actual_returns.flatten()
            else:
                actual_returns_flat = np.array(actual_returns).flatten()
        
        # Match lengths
        min_len = min(len(predictions_flat), len(actual_returns_flat))
        predictions_flat = predictions_flat[:min_len]
        actual_returns_flat = actual_returns_flat[:min_len]
        
        # Direction accuracy
        pred_directions = np.sign(predictions_flat)
        actual_directions = np.sign(actual_returns_flat)
        direction_accuracy = np.mean(pred_directions == actual_directions)
        
        print(f"• Direction accuracy: {direction_accuracy*100:.1f}%")
        
        # Correlation
        correlation = np.corrcoef(predictions_flat, actual_returns_flat)[0, 1]
        print(f"• Prediction correlation: {correlation:.4f}")
        
        # Trading signals analysis
        strong_buy_signals = np.sum(predictions_flat > 0.005)  # >0.5% predicted gain
        strong_sell_signals = np.sum(predictions_flat < -0.005)  # >0.5% predicted loss
        weak_signals = len(predictions_flat) - strong_buy_signals - strong_sell_signals
        
        print(f"• Strong BUY signals (>0.5%): {strong_buy_signals}")
        print(f"• Strong SELL signals (<-0.5%): {strong_sell_signals}")
        print(f"• Weak/HOLD signals: {weak_signals}")
        
        # Performance when confident
        strong_buy_mask = predictions_flat > 0.005
        strong_sell_mask = predictions_flat < -0.005
        
        if np.sum(strong_buy_mask) > 0:
            strong_buy_accuracy = np.mean(actual_returns_flat[strong_buy_mask] > 0)
            print(f"• Strong BUY accuracy: {strong_buy_accuracy*100:.1f}%")
        
        if np.sum(strong_sell_mask) > 0:
            strong_sell_accuracy = np.mean(actual_returns_flat[strong_sell_mask] < 0)
            print(f"• Strong SELL accuracy: {strong_sell_accuracy*100:.1f}%")
        
        # Sample recent predictions for trading decisions
        print(f"\n🔮 RECENT PREDICTIONS (for trading decisions):")
        recent_predictions = predictions[-10:]  # Last 10 hours
        recent_test_data = test_data.iloc[-10:]
        
        for i, (pred, (idx, row)) in enumerate(zip(recent_predictions, recent_test_data.iterrows())):
            predicted_return = float(pred)
            current_price = row['close']
            predicted_price = current_price * (1 + predicted_return)
            
            # Trading signal
            if predicted_return > 0.005:
                signal = "🟢 STRONG BUY"
                action = f"Target: ${predicted_price:.2f}"
            elif predicted_return > 0.001:
                signal = "🟡 WEAK BUY"
                action = f"Target: ${predicted_price:.2f}"
            elif predicted_return < -0.005:
                signal = "🔴 STRONG SELL"
                action = f"Target: ${predicted_price:.2f}"
            elif predicted_return < -0.001:
                signal = "🟠 WEAK SELL"
                action = f"Target: ${predicted_price:.2f}"
            else:
                signal = "⚪ HOLD"
                action = "No action"
            
            print(f"Hour -{10-i}: ${current_price:.2f} → {predicted_return*100:+.2f}% | {signal} | {action}")
        
        # Risk assessment
        max_predicted_loss = np.min(predictions) * 100
        max_predicted_gain = np.max(predictions) * 100
        print(f"\n⚠️  RISK ASSESSMENT:")
        print(f"• Max predicted loss: {max_predicted_loss:.2f}% per hour")
        print(f"• Max predicted gain: {max_predicted_gain:.2f}% per hour")
        print(f"• Model confidence range: ±{pred_std*100:.3f}%")
        
        # Final recommendation
        if direction_accuracy > 0.55 and correlation > 0.1:
            print(f"\n✅ MODEL READY FOR TRADING")
            print(f"   Direction accuracy: {direction_accuracy*100:.1f}% (>55% threshold)")
            print(f"   Correlation: {correlation:.3f} (>0.1 threshold)")
        else:
            print(f"\n❌ MODEL NEEDS IMPROVEMENT")
            print(f"   Direction accuracy: {direction_accuracy*100:.1f}% (<55% threshold)")
            print(f"   Correlation: {correlation:.3f} (<0.1 threshold)")
            print(f"   Recommend: More data, feature engineering, or architecture changes")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("The LSTM model failed - this needs debugging for production use")

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
        
        # Create more data for CNN (needs more samples)
        data = create_sample_data(days=30)
        
        # Simple config
        config = ChartPatternConfig(
            chart_height=16,     # Small chart for demo
            chart_width=32,
            epochs=2,            # Quick training
            batch_size=4,
            balance_classes=False,  # Disable class balancing for demo
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
        
        # Create data (need lots for technical indicators)
        data = create_sample_data(days=40)
        
        # Simple config
        config = MarketRegimeConfig(
            n_estimators=20,  # Small forest for demo
            max_depth=3,
            balance_classes=False,  # Disable class balancing for demo
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
    
    # Create data (need enough for meaningful backtest)
    data = create_sample_data(days=60)
    
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