#!/usr/bin/env python3
"""
Simple example of how to train models for the QuantumSentiment trading bot

This script demonstrates the basic workflow:
1. Train models on historical data
2. Save models with proper versioning
3. List available models
4. Test loading models
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from scripts.train_models import ModelTrainer


async def example_training():
    """Example of training models"""
    
    print("🚀 QuantumSentiment Model Training Example")
    print("=" * 50)
    
    # Configuration for training
    config = {
        'train_lstm': True,        # Train LSTM for price prediction
        'train_xgboost': True,     # Train XGBoost for market regime
        'train_ensemble': True,    # Train ensemble combining models
        'train_cnn': False,        # Skip CNN (requires more setup)
        'train_finbert': False,    # Skip FinBERT (requires heavy compute)
        'parallel_training': True, # Train models in parallel
        'max_workers': 4,          # Number of parallel workers
        'model_registry_path': 'model_registry',
        'model_save_dir': 'model_registry/saved_models'
    }
    
    # Initialize trainer
    print("📊 Initializing model trainer...")
    trainer = ModelTrainer(config)
    
    if not await trainer.initialize():
        print("❌ Failed to initialize trainer")
        return
    
    print("✅ Trainer initialized successfully")
    
    # Train on a small set of symbols with recent data
    symbols = ['AAPL', 'TSLA', 'GOOGL', 'MSFT']  # Start with 3 popular stocks
    days = 365  # Use 1 year of data
    
    print(f"📈 Loading training data for {symbols} ({days} days)...")
    
    try:
        training_data = await trainer.load_training_data(symbols, days)
        print(f"✅ Loaded {len(training_data):,} training samples")
        
        # Train models
        print("🧠 Training models...")
        results = await trainer.train_all_models(training_data)
        
        print("🎉 Training completed!")
        print(f"📦 Trained {results['total_models']} models")
        
        # Show results
        if results['saved_models']:
            print("\n📋 Trained Models:")
            for model_name, info in results['saved_models'].items():
                print(f"  • {model_name}:")
                print(f"    - Version: {info['version']}")
                print(f"    - Model ID: {info['model_id']}")
                
                # Show test metrics if available
                test_metrics = info.get('test_metrics', {})
                if test_metrics:
                    print(f"    - Test Metrics: {test_metrics}")
        
        # List all saved models
        print("\n📚 All Available Models:")
        saved_models = await trainer.list_saved_models()
        
        if saved_models:
            for model_name, versions in saved_models.items():
                print(f"  • {model_name}: {len(versions)} version(s)")
                for version in versions[:2]:  # Show latest 2 versions
                    print(f"    - {version}")
        else:
            print("  No models found in registry")
        
        print("\n✨ Success! Your models are now ready for trading.")
        print("💡 To use them, start the main trading loop:")
        print("   python src/main.py --mode paper")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(example_training())