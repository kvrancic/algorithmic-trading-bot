"""
Data module for QuantumSentiment Trading Bot

This module provides data fetching and processing capabilities
for various market data sources including stocks, crypto, and sentiment data.
"""

try:
    from .alpaca_client import AlpacaClient
    _has_alpaca = True
except ImportError:
    AlpacaClient = None
    _has_alpaca = False

try:
    from .data_fetcher import DataFetcher
    _has_data_fetcher = True
except ImportError:
    DataFetcher = None
    _has_data_fetcher = False

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

__all__ = []

# Add components if available
if _has_alpaca:
    __all__.append('AlpacaClient')
if _has_data_fetcher:
    __all__.append('DataFetcher')
if RedditClient:
    __all__.append('RedditClient')
if AlphaVantageClient:
    __all__.append('AlphaVantageClient')
if CryptoClient:
    __all__.append('CryptoClient')
if DataInterface:
    __all__.append('DataInterface')