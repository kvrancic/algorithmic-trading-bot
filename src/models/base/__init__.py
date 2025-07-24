"""
Base Model Classes for QuantumSentiment Trading Bot

Provides abstract base classes and interfaces for all ML models:
- BaseModel: Common interface for all models
- TimeSeriesModel: Base for LSTM and time series models
- ClassificationModel: Base for classification models
- EnsembleModel: Base for ensemble models
"""

from .base_model import BaseModel, ModelConfig, ModelType
from .time_series_model import TimeSeriesModel, TimeSeriesConfig
from .classification_model import ClassificationModel, ClassificationConfig
from .ensemble_model import EnsembleModel, EnsembleConfig

__all__ = [
    'BaseModel', 'ModelConfig', 'ModelType',
    'TimeSeriesModel', 'TimeSeriesConfig',
    'ClassificationModel', 'ClassificationConfig', 
    'EnsembleModel', 'EnsembleConfig'
]