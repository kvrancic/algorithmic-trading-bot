"""
QuantumSentiment Trading Bot - Machine Learning Models

This module contains all ML models for the trading system:
- PriceLSTM: LSTM model for price prediction
- ChartPatternCNN: CNN for chart pattern recognition
- MarketRegimeXGBoost: XGBoost for market regime classification
- FinBERT: Transformer model for sentiment analysis
- StackedEnsemble: Meta-learner combining all models
"""

# Base classes
from .base import (
    BaseModel, ModelConfig, ModelType,
    TimeSeriesModel, TimeSeriesConfig,
    ClassificationModel, ClassificationConfig,
    EnsembleModel, EnsembleConfig
)

# Import specific models
from .lstm import PriceLSTM, PriceLSTMConfig
from .cnn import ChartPatternCNN, ChartPatternConfig
from .xgboost import MarketRegimeXGBoost, MarketRegimeConfig
from .transformers import FinBERT, FinBERTConfig
from .ensemble import StackedEnsemble, StackedEnsembleConfig

__all__ = [
    # Base classes
    'BaseModel', 'ModelConfig', 'ModelType',
    'TimeSeriesModel', 'TimeSeriesConfig',
    'ClassificationModel', 'ClassificationConfig',
    'EnsembleModel', 'EnsembleConfig',
    
    # Specific models
    'PriceLSTM', 'PriceLSTMConfig',
    'ChartPatternCNN', 'ChartPatternConfig',
    'MarketRegimeXGBoost', 'MarketRegimeConfig',
    'FinBERT', 'FinBERTConfig',
    'StackedEnsemble', 'StackedEnsembleConfig'
]