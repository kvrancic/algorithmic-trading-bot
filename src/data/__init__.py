"""
Data module for QuantumSentiment Trading Bot

This module provides data fetching and processing capabilities
for various market data sources including stocks, crypto, and sentiment data.
"""

from .alpaca_client import AlpacaClient
from .reddit_client import RedditClient
from .alpha_vantage_client import AlphaVantageClient
from .crypto_client import CryptoClient
from .data_interface import DataInterface

__all__ = [
    'AlpacaClient',
    'RedditClient', 
    'AlphaVantageClient',
    'CryptoClient',
    'DataInterface'
]