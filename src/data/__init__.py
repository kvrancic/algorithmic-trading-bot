"""
Data module for QuantumSentiment Trading Bot

This module provides data fetching and processing capabilities
for various market data sources including stocks, crypto, and sentiment data.
"""

from .alpaca_client import AlpacaClient
from .data_fetcher import DataFetcher

try:
    from .reddit_client import RedditClient
    from .alpha_vantage_client import AlphaVantageClient
    from .crypto_client import CryptoClient
    from .data_interface import DataInterface
except ImportError:
    # Optional components
    RedditClient = None
    AlphaVantageClient = None
    CryptoClient = None
    DataInterface = None

__all__ = [
    'AlpacaClient',
    'DataFetcher'
]

# Add optional components if available
if RedditClient:
    __all__.append('RedditClient')
if AlphaVantageClient:
    __all__.append('AlphaVantageClient')
if CryptoClient:
    __all__.append('CryptoClient')
if DataInterface:
    __all__.append('DataInterface')