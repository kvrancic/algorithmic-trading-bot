"""
Multi-Source Sentiment Analysis Package

This package provides comprehensive sentiment analysis capabilities for the
QuantumSentiment trading bot, including:

- Reddit sentiment analysis (praw library)
- Twitter sentiment analysis (optional API)
- UnusualWhales political intelligence (web scraping)
- News aggregation and sentiment analysis
- Multi-source sentiment fusion algorithms

All components are designed for real-time trading signal generation.
"""

from .reddit_analyzer import RedditSentimentAnalyzer, RedditConfig
from .news_aggregator import NewsAggregator, NewsConfig
from .unusual_whales_analyzer import UnusualWhalesAnalyzer, UnusualWhalesConfig
from .sentiment_fusion import SentimentFusion, FusionConfig

__all__ = [
    'RedditSentimentAnalyzer', 'RedditConfig',
    'NewsAggregator', 'NewsConfig',
    'UnusualWhalesAnalyzer', 'UnusualWhalesConfig',
    'SentimentFusion', 'FusionConfig'
]

__version__ = "1.0.0"