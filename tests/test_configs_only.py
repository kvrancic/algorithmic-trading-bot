"""
Test just the config creation to make sure they all work
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pathlib import Path

def test_all_configs():
    """Test that all model configs can be created"""
    print("Testing all model configurations...")
    
    try:
        # Test LSTM config
        from models.lstm.price_lstm import PriceLSTMConfig
        lstm_config = PriceLSTMConfig(
            sequence_length=10,
            epochs=2,
            batch_size=4,
            save_path=Path("test_output")
        )
        print(f"✓ PriceLSTMConfig: {lstm_config.name}")
        
        # Test CNN config
        from models.cnn.chart_pattern_cnn import ChartPatternConfig
        cnn_config = ChartPatternConfig(
            epochs=2,
            batch_size=2,
            chart_height=32,
            chart_width=32,
            save_path=Path("test_output")
        )
        print(f"✓ ChartPatternConfig: {cnn_config.name}")
        
        # Test XGBoost config
        from models.xgboost.market_regime_xgboost import MarketRegimeConfig
        xgb_config = MarketRegimeConfig(
            n_estimators=10,
            epochs=2,
            save_path=Path("test_output")
        )
        print(f"✓ MarketRegimeConfig: {xgb_config.name}")
        
        # Test FinBERT config
        from models.transformers.finbert import FinBERTConfig
        bert_config = FinBERTConfig(
            epochs=1,
            batch_size=2,
            save_path=Path("test_output")
        )
        print(f"✓ FinBERTConfig: {bert_config.name}")
        
        # Test Ensemble config
        from models.ensemble.stacked_ensemble import StackedEnsembleConfig
        ensemble_config = StackedEnsembleConfig(
            save_path=Path("test_output")
        )
        print(f"✓ StackedEnsembleConfig: {ensemble_config.name}")
        
        print("\n🎉 All configs created successfully!")
        return True
    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_all_configs()
    exit(0 if success else 1)