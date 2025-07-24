"""
Database Models for QuantumSentiment Trading Bot

SQLAlchemy models for storing all trading-related data:
- Market data (OHLCV, quotes)
- Sentiment data (Reddit, Twitter, news)
- Fundamental data (financials, ratios)
- Trading signals and predictions
- Performance metrics
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, JSON,
    Index, ForeignKey, UniqueConstraint, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()


class MarketData(Base):
    """Market data table for OHLCV data"""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)  # 1min, 5min, 1h, 1d, etc.
    
    # OHLCV data
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False, default=0)
    
    # Additional market data
    vwap = Column(Float)  # Volume weighted average price
    trades_count = Column(Integer)
    
    # Data source and quality
    source = Column(String(50), nullable=False)  # alpaca, alphavantage, binance, etc.
    quality_score = Column(Float, default=1.0)  # Data quality indicator
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('symbol', 'timestamp', 'timeframe', 'source', name='unique_market_data'),
        CheckConstraint('open > 0', name='check_positive_open'),
        CheckConstraint('high > 0', name='check_positive_high'),
        CheckConstraint('low > 0', name='check_positive_low'),
        CheckConstraint('close > 0', name='check_positive_close'),
        CheckConstraint('volume >= 0', name='check_non_negative_volume'),
        CheckConstraint('high >= low', name='check_high_low_relationship'),
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_timeframe_timestamp', 'timeframe', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<MarketData(symbol='{self.symbol}', timestamp='{self.timestamp}', close={self.close})>"


class SentimentData(Base):
    """Sentiment data from various sources"""
    __tablename__ = 'sentiment_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    source = Column(String(50), nullable=False)  # reddit, twitter, news, etc.
    
    # Sentiment metrics
    sentiment_score = Column(Float, nullable=False)  # -1 to 1
    confidence = Column(Float, nullable=False)  # 0 to 1
    mention_count = Column(Integer, default=0)
    
    # Source-specific metrics
    bullish_signals = Column(Integer, default=0)
    bearish_signals = Column(Integer, default=0)
    neutral_signals = Column(Integer, default=0)
    
    # Engagement metrics
    total_engagement = Column(Integer, default=0)  # likes, shares, comments
    avg_engagement = Column(Float, default=0.0)
    
    # Content analysis
    keywords = Column(JSON)  # Extracted keywords/topics
    emotions = Column(JSON)  # Emotional analysis if available
    
    # Data quality and metadata
    quality_score = Column(Float, default=1.0)
    raw_data = Column(JSON)  # Store original data for debugging
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('symbol', 'timestamp', 'source', name='unique_sentiment_data'),
        CheckConstraint('sentiment_score >= -1 AND sentiment_score <= 1', name='check_sentiment_range'),
        CheckConstraint('confidence >= 0 AND confidence <= 1', name='check_confidence_range'),
        CheckConstraint('mention_count >= 0', name='check_positive_mentions'),
        Index('idx_sentiment_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_sentiment_score', 'sentiment_score'),
    )
    
    def __repr__(self):
        return f"<SentimentData(symbol='{self.symbol}', source='{self.source}', sentiment={self.sentiment_score})>"


class FundamentalData(Base):
    """Fundamental data for stocks"""
    __tablename__ = 'fundamental_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    period = Column(String(20), nullable=False)  # quarterly, annual, ttm
    
    # Company info
    company_name = Column(String(200))
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(Float)
    
    # Valuation metrics
    pe_ratio = Column(Float)
    peg_ratio = Column(Float)
    pb_ratio = Column(Float)
    ps_ratio = Column(Float)
    ev_ebitda = Column(Float)
    
    # Profitability metrics
    roe = Column(Float)  # Return on Equity
    roa = Column(Float)  # Return on Assets
    profit_margin = Column(Float)
    operating_margin = Column(Float)
    gross_margin = Column(Float)
    
    # Financial health
    debt_to_equity = Column(Float)
    current_ratio = Column(Float)
    quick_ratio = Column(Float)
    cash_ratio = Column(Float)
    
    # Growth metrics
    revenue_growth = Column(Float)
    earnings_growth = Column(Float)
    dividend_yield = Column(Float)
    
    # Per share metrics
    eps = Column(Float)  # Earnings per share
    book_value_per_share = Column(Float)
    revenue_per_share = Column(Float)
    
    # Raw financial data
    revenue = Column(Float)
    net_income = Column(Float)
    total_assets = Column(Float)
    total_debt = Column(Float)
    cash_and_equivalents = Column(Float)
    
    # Additional data
    beta = Column(Float)  # Market beta
    shares_outstanding = Column(Float)
    analyst_target_price = Column(Float)
    
    source = Column(String(50), nullable=False, default='alphavantage')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('symbol', 'period', 'timestamp', name='unique_fundamental_data'),
        Index('idx_fundamental_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_pe_ratio', 'pe_ratio'),
        Index('idx_market_cap', 'market_cap'),
    )
    
    def __repr__(self):
        return f"<FundamentalData(symbol='{self.symbol}', period='{self.period}', pe_ratio={self.pe_ratio})>"


class TradingSignal(Base):
    """Trading signals and ML predictions"""
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String(50), unique=True, default=lambda: str(uuid.uuid4()))
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Signal details
    signal_type = Column(String(50), nullable=False)  # buy, sell, hold
    signal_strength = Column(Float, nullable=False)  # 0 to 1
    confidence = Column(Float, nullable=False)  # 0 to 1
    
    # Price predictions
    target_price = Column(Float)
    stop_loss_price = Column(Float)
    expected_return = Column(Float)
    risk_score = Column(Float)
    
    # Model information
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50))
    features_used = Column(JSON)  # List of features used
    
    # Signal components
    technical_score = Column(Float, default=0.0)
    fundamental_score = Column(Float, default=0.0)
    sentiment_score = Column(Float, default=0.0)
    
    # Execution details
    executed = Column(Boolean, default=False)
    execution_price = Column(Float)
    execution_timestamp = Column(DateTime)
    
    # Performance tracking
    actual_return = Column(Float)
    performance_measured = Column(Boolean, default=False)
    measurement_timestamp = Column(DateTime)
    
    # Metadata
    timeframe = Column(String(20), nullable=False)  # Prediction timeframe
    expires_at = Column(DateTime)  # When signal expires
    notes = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        CheckConstraint('signal_strength >= 0 AND signal_strength <= 1', name='check_signal_strength_range'),
        CheckConstraint('confidence >= 0 AND confidence <= 1', name='check_signal_confidence_range'),
        Index('idx_signal_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_signal_type', 'signal_type'),
        Index('idx_model_name', 'model_name'),
        Index('idx_executed', 'executed'),
    )
    
    def __repr__(self):
        return f"<TradingSignal(symbol='{self.symbol}', signal_type='{self.signal_type}', strength={self.signal_strength})>"


class FeatureData(Base):
    """Engineered features for ML models"""
    __tablename__ = 'feature_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    
    # Feature categories
    technical_features = Column(JSON)  # Technical indicators
    sentiment_features = Column(JSON)  # Sentiment metrics
    fundamental_features = Column(JSON)  # Fundamental ratios
    market_structure_features = Column(JSON)  # Market microstructure
    macro_features = Column(JSON)  # Macroeconomic indicators
    
    # Feature metadata
    feature_version = Column(String(50), nullable=False)
    total_features = Column(Integer, nullable=False)
    missing_features = Column(Integer, default=0)
    quality_score = Column(Float, default=1.0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('symbol', 'timestamp', 'timeframe', 'feature_version', name='unique_feature_data'),
        Index('idx_feature_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_feature_version', 'feature_version'),
    )
    
    def __repr__(self):
        return f"<FeatureData(symbol='{self.symbol}', timestamp='{self.timestamp}', features={self.total_features})>"


class PerformanceMetrics(Base):
    """Performance tracking for strategies and models"""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(100), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    
    # Returns
    daily_return = Column(Float, nullable=False)
    cumulative_return = Column(Float, nullable=False)
    benchmark_return = Column(Float)  # SPY or other benchmark
    
    # Risk metrics
    volatility = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    var_95 = Column(Float)  # 95% Value at Risk
    cvar_95 = Column(Float)  # 95% Conditional VaR
    
    # Trading metrics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float)
    avg_win = Column(Float)
    avg_loss = Column(Float)
    profit_factor = Column(Float)
    
    # Portfolio metrics
    portfolio_value = Column(Float, nullable=False)
    cash_balance = Column(Float)
    positions_count = Column(Integer, default=0)
    leverage = Column(Float, default=1.0)
    
    # Model performance (if applicable)
    prediction_accuracy = Column(Float)
    signal_count = Column(Integer, default=0)
    signal_accuracy = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('strategy_name', 'date', name='unique_performance_metrics'),
        Index('idx_performance_strategy_date', 'strategy_name', 'date'),
        Index('idx_sharpe_ratio', 'sharpe_ratio'),
        Index('idx_max_drawdown', 'max_drawdown'),
    )
    
    def __repr__(self):
        return f"<PerformanceMetrics(strategy='{self.strategy_name}', return={self.daily_return}, sharpe={self.sharpe_ratio})>"


class EconomicData(Base):
    """Economic indicators and macro data"""
    __tablename__ = 'economic_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    indicator = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Economic data
    value = Column(Float, nullable=False)
    unit = Column(String(50))  # %, dollars, index, etc.
    frequency = Column(String(20))  # daily, monthly, quarterly
    
    # Metadata
    source = Column(String(50), nullable=False)
    release_date = Column(DateTime)
    revision = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('indicator', 'timestamp', 'source', name='unique_economic_data'),
        Index('idx_economic_indicator_timestamp', 'indicator', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<EconomicData(indicator='{self.indicator}', value={self.value})>"


class NewsData(Base):
    """News articles and their sentiment analysis"""
    __tablename__ = 'news_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    article_id = Column(String(100), unique=True, nullable=False)
    
    # Article details
    title = Column(Text, nullable=False)
    summary = Column(Text)
    url = Column(String(500))
    source = Column(String(100), nullable=False)
    author = Column(String(200))
    
    # Timing
    published_at = Column(DateTime, nullable=False, index=True)
    scraped_at = Column(DateTime, default=datetime.utcnow)
    
    # Content analysis
    symbols_mentioned = Column(JSON)  # List of symbols mentioned
    topics = Column(JSON)  # Topic classification
    sentiment_score = Column(Float)  # Overall sentiment
    relevance_score = Column(Float)  # Relevance to trading
    
    # Engagement (if available)
    social_shares = Column(Integer, default=0)
    comments_count = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_news_published_at', 'published_at'),
        Index('idx_news_source', 'source'),
        Index('idx_news_sentiment', 'sentiment_score'),
    )
    
    def __repr__(self):
        return f"<NewsData(title='{self.title[:50]}...', source='{self.source}')>"