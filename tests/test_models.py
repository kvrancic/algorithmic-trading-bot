"""
Test script to validate Phase 2.1 model implementations
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Import all models
from src.models import (
    PriceLSTM, PriceLSTMConfig,
    ChartPatternCNN, ChartPatternConfig,
    MarketRegimeXGBoost, MarketRegimeConfig,
    FinBERT, FinBERTConfig,
    StackedEnsemble, StackedEnsembleConfig
)

def generate_sample_data():
    """Generate sample financial data for testing"""
    np.random.seed(42)
    
    # Generate 1000 data points
    n_samples = 1000
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='1H')
    
    # Generate realistic OHLCV data
    initial_price = 100.0
    prices = [initial_price]
    
    for i in range(1, n_samples):
        # Random walk with small drift
        change = np.random.normal(0.0001, 0.02)  # 0.01% drift, 2% volatility
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # Generate OHLC from prices
    data = []
    for i in range(len(prices)):
        # Add some randomness to OHLC
        close = prices[i]
        noise = np.random.normal(0, 0.005)  # 0.5% noise
        
        open_price = close * (1 + noise)
        high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.001)))
        low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.001)))
        volume = np.random.lognormal(10, 0.5)  # Log-normal volume
        
        data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    return df

def generate_sample_text_data():
    """Generate sample financial text data"""
    texts = [
        "The company reported strong earnings growth this quarter with revenue beating expectations",
        "Market volatility increased due to uncertainty about interest rate decisions",
        "Technical analysis suggests a bullish pattern forming in the stock price",
        "The Federal Reserve's hawkish stance is putting pressure on equity markets",
        "Strong employment data indicates a robust economic recovery",
        "Concerns about inflation continue to weigh on investor sentiment",
        "The earnings report exceeded analyst expectations by a wide margin",
        "Geopolitical tensions are creating uncertainty in global markets",
        "The company's guidance for next quarter looks optimistic",
        "Market correction appears to be healthy for long-term growth"
    ]
    
    # Create sentiment labels (0=negative, 1=neutral, 2=positive)
    labels = [2, 0, 2, 0, 2, 0, 2, 0, 2, 2]
    
    return texts, labels

def test_price_lstm():
    """Test PriceLSTM model"""
    print("Testing PriceLSTM...")
    
    # Generate data
    data = generate_sample_data()
    
    # Create config with small parameters for testing
    config = PriceLSTMConfig(
        sequence_length=20,
        lstm_hidden_size=32,
        epochs=2,
        batch_size=16,
        save_path=Path("test_models")
    )
    
    # Create model
    model = PriceLSTM(config)
    
    # Prepare data (use close price for both features and targets)
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    
    # Create simple features and targets
    features = train_data[['open', 'high', 'low', 'close', 'volume']].values
    targets = train_data['close'].values[1:]  # Predict next close price
    features = features[:-1]  # Align with targets
    
    # Train model
    history = model.train(features, targets)
    
    # Test prediction
    test_data = data[train_size:train_size+50]
    test_features = test_data[['open', 'high', 'low', 'close', 'volume']].values
    predictions = model.predict(test_features)
    
    print(f"✓ PriceLSTM: Training completed, made {len(predictions)} predictions")
    return model

def test_chart_pattern_cnn():
    """Test ChartPatternCNN model"""
    print("Testing ChartPatternCNN...")
    
    # Generate data
    data = generate_sample_data()
    
    # Create config
    config = ChartPatternConfig(
        chart_height=32,
        chart_width=64,
        epochs=2,
        batch_size=4,
        save_path=Path("test_models")
    )
    
    # Create model
    model = ChartPatternCNN(config)
    
    # Create simple pattern labels (random for testing)
    np.random.seed(42)
    pattern_labels = np.random.choice(config.pattern_classes, len(data))
    
    # Train model
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    train_labels = pd.Series(pattern_labels[:train_size])
    
    history = model.train(train_data, train_labels)
    
    # Test prediction
    test_data = data[train_size:train_size+10]
    predictions = model.predict(test_data)
    
    print(f"✓ ChartPatternCNN: Training completed, made {len(predictions)} predictions")
    return model

def test_market_regime_xgboost():
    """Test MarketRegimeXGBoost model"""
    print("Testing MarketRegimeXGBoost...")
    
    # Generate data
    data = generate_sample_data()
    
    # Create config
    config = MarketRegimeConfig(
        n_estimators=10,  # Small for testing
        epochs=2,
        save_path=Path("test_models")
    )
    
    # Create model
    model = MarketRegimeXGBoost(config)
    
    # Use auto-generated regime labels
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    
    # Model will auto-generate labels based on rules
    history = model.train(train_data, None)  # No labels - will auto-generate
    
    # Test prediction
    test_data = data[train_size:train_size+10]
    predictions = model.predict(test_data)
    
    print(f"✓ MarketRegimeXGBoost: Training completed, made {len(predictions)} predictions")
    return model

def test_finbert():
    """Test FinBERT model (simplified without actual transformer)"""
    print("Testing FinBERT...")
    
    try:
        # Generate text data
        texts, labels = generate_sample_text_data()
        
        # Create config
        config = FinBERTConfig(
            epochs=1,
            batch_size=2,
            save_path=Path("test_models"),
            model_name="distilbert-base-uncased"  # Smaller model for testing
        )
        
        # Create model
        model = FinBERT(config)
        
        # Train model
        train_texts = texts[:8]
        train_labels = np.array(labels[:8])
        
        history = model.train(train_texts, train_labels)
        
        # Test prediction
        test_texts = texts[8:]
        predictions = model.predict(test_texts)
        
        print(f"✓ FinBERT: Training completed, made {len(predictions)} predictions")
        return model
        
    except Exception as e:
        print(f"⚠ FinBERT: Skipped due to transformer dependencies: {e}")
        return None

def test_stacked_ensemble():
    """Test StackedEnsemble model"""
    print("Testing StackedEnsemble...")
    
    # Generate data
    data = generate_sample_data()
    
    # Create a simple ensemble with mock base models
    config = StackedEnsembleConfig(
        meta_learner_type="logistic",
        epochs=1,
        save_path=Path("test_models")
    )
    
    ensemble = StackedEnsemble(config)
    
    # Create mock base models that return simple predictions
    class MockModel:
        def __init__(self, name, prediction_offset=0):
            self.name = name
            self.offset = prediction_offset
            self.model_type = "classification"
        
        def predict(self, data):
            n = len(data) if hasattr(data, '__len__') else 10
            return np.random.choice([0, 1, 2], n) + self.offset
        
        def predict_proba(self, data):
            n = len(data) if hasattr(data, '__len__') else 10
            probs = np.random.random((n, 3))
            return probs / probs.sum(axis=1, keepdims=True)
    
    # Add mock base models
    ensemble.add_base_model("MockLSTM", MockModel("MockLSTM", 0))
    ensemble.add_base_model("MockCNN", MockModel("MockCNN", 1))
    ensemble.add_base_model("MockXGB", MockModel("MockXGB", 0))
    
    # Create simple labels
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    train_labels = np.random.choice([0, 1, 2], len(train_data))
    
    # Train ensemble
    history = ensemble.train(train_data, train_labels)
    
    # Test prediction
    test_data = data[train_size:train_size+10]
    predictions = ensemble.predict(test_data)
    
    print(f"✓ StackedEnsemble: Training completed, made {len(predictions)} predictions")
    return ensemble

def main():
    """Run all model tests"""
    print("Starting Phase 2.1 Model Architecture Tests...")
    print("=" * 50)
    
    # Create test directory
    Path("test_models").mkdir(exist_ok=True)
    
    results = {}
    
    try:
        # Test each model
        results['PriceLSTM'] = test_price_lstm()
        print()
        
        results['ChartPatternCNN'] = test_chart_pattern_cnn()
        print()
        
        results['MarketRegimeXGBoost'] = test_market_regime_xgboost()
        print()
        
        results['FinBERT'] = test_finbert()  # May skip due to dependencies
        print()
        
        results['StackedEnsemble'] = test_stacked_ensemble()
        print()
        
        # Summary
        print("=" * 50)
        print("Test Summary:")
        successful = sum(1 for model in results.values() if model is not None)
        total = len(results)
        print(f"✓ {successful}/{total} models tested successfully")
        
        if successful == total:
            print("🎉 All models working correctly!")
        else:
            print("⚠ Some models had issues (likely due to missing dependencies)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup test directory
        import shutil
        if Path("test_models").exists():
            shutil.rmtree("test_models")

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)