"""
Feature Engineering Module for QuantumSentiment Trading Bot

Comprehensive feature engineering pipeline including:
- Technical indicators (100+ features)
- Sentiment features
- Fundamental features  
- Market microstructure features
- Macroeconomic features
"""

from .technical import TechnicalFeatures
from .sentiment import SentimentFeatures
from .feature_pipeline import FeaturePipeline, FeatureConfig

# Import optional features (will be implemented in Phase 2)
try:
    from .fundamental import FundamentalFeatures
except ImportError:
    FundamentalFeatures = None

try:
    from .market_structure import MarketStructureFeatures  
except ImportError:
    MarketStructureFeatures = None

try:
    from .macro import MacroFeatures
except ImportError:
    MacroFeatures = None

__all__ = [
    'TechnicalFeatures',
    'SentimentFeatures', 
    'FeaturePipeline',
    'FeatureConfig'
]

# Add optional features if available
if FundamentalFeatures:
    __all__.append('FundamentalFeatures')
if MarketStructureFeatures:
    __all__.append('MarketStructureFeatures')
if MacroFeatures:
    __all__.append('MacroFeatures')