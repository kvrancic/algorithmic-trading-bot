"""
Database Manager for QuantumSentiment Trading Bot

Handles database connections, session management, and high-level operations
for PostgreSQL and SQLite databases.
"""

import os
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timedelta
from contextlib import contextmanager
import pandas as pd
import structlog
from sqlalchemy import create_engine, text, func, and_, or_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import SQLAlchemyError

from .models import (
    Base, MarketData, SentimentData, FundamentalData, 
    TradingSignal, FeatureData, PerformanceMetrics,
    EconomicData, NewsData
)

logger = structlog.get_logger(__name__)


class DatabaseManager:
    """High-level database manager for all trading data"""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database manager
        
        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url or os.getenv(
            'DATABASE_URL', 
            'sqlite:///data/quantum.db'
        )
        
        # Configure engine based on database type
        if self.database_url.startswith('sqlite'):
            self.engine = create_engine(
                self.database_url,
                poolclass=StaticPool,
                connect_args={
                    'check_same_thread': False,
                    'timeout': 30
                },
                echo=False
            )
        else:
            # PostgreSQL or other databases
            self.engine = create_engine(
                self.database_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=False
            )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        self.db_type = 'sqlite' if 'sqlite' in self.database_url else 'postgresql'
        
        logger.info("Database manager initialized", 
                   db_type=self.db_type,
                   url=self.database_url.split('@')[-1] if '@' in self.database_url else self.database_url)
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error("Failed to create database tables", error=str(e))
            raise
    
    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error("Failed to drop database tables", error=str(e))
            raise
    
    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Database session error", error=str(e))
            raise
        finally:
            session.close()
    
    # === MARKET DATA OPERATIONS ===
    
    def save_market_data(
        self,
        symbol: str,
        timestamp: datetime,
        timeframe: str,
        ohlcv_data: Dict[str, float],
        source: str = 'unknown',
        **kwargs
    ) -> bool:
        """
        Save market data to database
        
        Args:
            symbol: Asset symbol
            timestamp: Data timestamp
            timeframe: Data timeframe (1min, 1h, 1d, etc.)
            ohlcv_data: Dictionary with OHLCV data
            source: Data source
            **kwargs: Additional data fields
            
        Returns:
            Success status
        """
        try:
            with self.get_session() as session:
                market_data = MarketData(
                    symbol=symbol.upper(),
                    timestamp=timestamp,
                    timeframe=timeframe,
                    open=ohlcv_data['open'],
                    high=ohlcv_data['high'],
                    low=ohlcv_data['low'],
                    close=ohlcv_data['close'],
                    volume=ohlcv_data.get('volume', 0),
                    vwap=kwargs.get('vwap'),
                    trades_count=kwargs.get('trades_count'),
                    source=source,
                    quality_score=kwargs.get('quality_score', 1.0)
                )
                
                session.merge(market_data)  # Use merge to handle duplicates
                
            logger.debug("Market data saved", 
                        symbol=symbol, 
                        timestamp=timestamp,
                        source=source)
            return True
            
        except Exception as e:
            logger.error("Failed to save market data", 
                        symbol=symbol, error=str(e))
            return False
    
    def get_market_data(
        self,
        symbol: str,
        timeframe: str = '1h',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get market data from database
        
        Args:
            symbol: Asset symbol
            timeframe: Data timeframe
            start_date: Start date for data
            end_date: End date for data
            limit: Maximum number of records
            
        Returns:
            DataFrame with market data
        """
        try:
            with self.get_session() as session:
                query = session.query(MarketData).filter(
                    MarketData.symbol == symbol.upper(),
                    MarketData.timeframe == timeframe
                )
                
                if start_date:
                    query = query.filter(MarketData.timestamp >= start_date)
                if end_date:
                    query = query.filter(MarketData.timestamp <= end_date)
                
                query = query.order_by(MarketData.timestamp.desc())
                
                if limit:
                    query = query.limit(limit)
                
                results = query.all()
                
                if not results:
                    return pd.DataFrame()
                
                # Convert to DataFrame
                data = []
                for row in results:
                    data.append({
                        'timestamp': row.timestamp,
                        'open': row.open,
                        'high': row.high,
                        'low': row.low,
                        'close': row.close,
                        'volume': row.volume,
                        'vwap': row.vwap,
                        'source': row.source
                    })
                
                df = pd.DataFrame(data)
                df.set_index('timestamp', inplace=True)
                df = df.sort_index()
                
                logger.debug("Market data retrieved", 
                           symbol=symbol, 
                           timeframe=timeframe,
                           records=len(df))
                
                return df
                
        except Exception as e:
            logger.error("Failed to get market data", 
                        symbol=symbol, error=str(e))
            return pd.DataFrame()
    
    def save_market_data_batch(
        self,
        data_list: List[Dict[str, Any]]
    ) -> int:
        """Save multiple market data records efficiently"""
        try:
            with self.get_session() as session:
                market_data_objects = []
                
                for data in data_list:
                    market_data = MarketData(**data)
                    market_data_objects.append(market_data)
                
                # Use bulk insert for better performance
                session.bulk_save_objects(market_data_objects)
                
                logger.info("Market data batch saved", count=len(data_list))
                return len(data_list)
                
        except Exception as e:
            logger.error("Failed to save market data batch", error=str(e))
            return 0
    
    # === SENTIMENT DATA OPERATIONS ===
    
    def save_sentiment_data(
        self,
        symbol: str,
        timestamp: datetime,
        source: str,
        sentiment_score: float,
        confidence: float,
        **kwargs
    ) -> bool:
        """Save sentiment data to database"""
        try:
            with self.get_session() as session:
                sentiment_data = SentimentData(
                    symbol=symbol.upper(),
                    timestamp=timestamp,
                    source=source,
                    sentiment_score=sentiment_score,
                    confidence=confidence,
                    mention_count=kwargs.get('mention_count', 0),
                    bullish_signals=kwargs.get('bullish_signals', 0),
                    bearish_signals=kwargs.get('bearish_signals', 0),
                    neutral_signals=kwargs.get('neutral_signals', 0),
                    total_engagement=kwargs.get('total_engagement', 0),
                    avg_engagement=kwargs.get('avg_engagement', 0.0),
                    keywords=kwargs.get('keywords'),
                    emotions=kwargs.get('emotions'),
                    quality_score=kwargs.get('quality_score', 1.0),
                    raw_data=kwargs.get('raw_data')
                )
                
                session.merge(sentiment_data)
                
            logger.debug("Sentiment data saved", 
                        symbol=symbol, 
                        source=source,
                        sentiment=sentiment_score)
            return True
            
        except Exception as e:
            logger.error("Failed to save sentiment data", 
                        symbol=symbol, error=str(e))
            return False
    
    def get_sentiment_data(
        self,
        symbol: str,
        sources: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Get sentiment data from database"""
        try:
            with self.get_session() as session:
                query = session.query(SentimentData).filter(
                    SentimentData.symbol == symbol.upper()
                )
                
                if sources:
                    query = query.filter(SentimentData.source.in_(sources))
                if start_date:
                    query = query.filter(SentimentData.timestamp >= start_date)
                if end_date:
                    query = query.filter(SentimentData.timestamp <= end_date)
                
                query = query.order_by(SentimentData.timestamp.desc())
                
                if limit:
                    query = query.limit(limit)
                
                results = query.all()
                
                if not results:
                    return pd.DataFrame()
                
                # Convert to DataFrame
                data = []
                for row in results:
                    data.append({
                        'timestamp': row.timestamp,
                        'source': row.source,
                        'sentiment_score': row.sentiment_score,
                        'confidence': row.confidence,
                        'mention_count': row.mention_count,
                        'bullish_signals': row.bullish_signals,
                        'bearish_signals': row.bearish_signals
                    })
                
                df = pd.DataFrame(data)
                if not df.empty:
                    df.set_index('timestamp', inplace=True)
                    df = df.sort_index()
                
                logger.debug("Sentiment data retrieved", 
                           symbol=symbol, 
                           records=len(df))
                
                return df
                
        except Exception as e:
            logger.error("Failed to get sentiment data", 
                        symbol=symbol, error=str(e))
            return pd.DataFrame()
    
    # === TRADING SIGNALS OPERATIONS ===
    
    def save_trading_signal(
        self,
        symbol: str,
        timestamp: datetime,
        signal_type: str,
        signal_strength: float,
        confidence: float,
        model_name: str,
        **kwargs
    ) -> str:
        """Save trading signal and return signal ID"""
        try:
            with self.get_session() as session:
                signal = TradingSignal(
                    symbol=symbol.upper(),
                    timestamp=timestamp,
                    signal_type=signal_type,
                    signal_strength=signal_strength,
                    confidence=confidence,
                    target_price=kwargs.get('target_price'),
                    stop_loss_price=kwargs.get('stop_loss_price'),
                    expected_return=kwargs.get('expected_return'),
                    risk_score=kwargs.get('risk_score'),
                    model_name=model_name,
                    model_version=kwargs.get('model_version'),
                    features_used=kwargs.get('features_used'),
                    technical_score=kwargs.get('technical_score', 0.0),
                    fundamental_score=kwargs.get('fundamental_score', 0.0),
                    sentiment_score=kwargs.get('sentiment_score', 0.0),
                    timeframe=kwargs.get('timeframe', '1d'),
                    expires_at=kwargs.get('expires_at'),
                    notes=kwargs.get('notes')
                )
                
                session.add(signal)
                session.flush()  # Get the signal_id
                signal_id = signal.signal_id
                
            logger.info("Trading signal saved", 
                       symbol=symbol, 
                       signal_type=signal_type,
                       signal_id=signal_id)
            
            return signal_id
            
        except Exception as e:
            logger.error("Failed to save trading signal", 
                        symbol=symbol, error=str(e))
            return ""
    
    def get_active_signals(
        self,
        symbol: Optional[str] = None,
        signal_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get active (non-expired, non-executed) trading signals"""
        try:
            with self.get_session() as session:
                query = session.query(TradingSignal).filter(
                    TradingSignal.executed == False,
                    or_(
                        TradingSignal.expires_at.is_(None),
                        TradingSignal.expires_at > datetime.utcnow()
                    )
                )
                
                if symbol:
                    query = query.filter(TradingSignal.symbol == symbol.upper())
                if signal_types:
                    query = query.filter(TradingSignal.signal_type.in_(signal_types))
                
                results = query.order_by(TradingSignal.timestamp.desc()).all()
                
                signals = []
                for signal in results:
                    signals.append({
                        'signal_id': signal.signal_id,
                        'symbol': signal.symbol,
                        'timestamp': signal.timestamp,
                        'signal_type': signal.signal_type,
                        'signal_strength': signal.signal_strength,
                        'confidence': signal.confidence,
                        'target_price': signal.target_price,
                        'stop_loss_price': signal.stop_loss_price,
                        'model_name': signal.model_name,
                        'expires_at': signal.expires_at
                    })
                
                logger.debug("Active signals retrieved", count=len(signals))
                return signals
                
        except Exception as e:
            logger.error("Failed to get active signals", error=str(e))
            return []
    
    def update_signal_execution(
        self,
        signal_id: str,
        execution_price: float,
        execution_timestamp: datetime
    ) -> bool:
        """Update signal with execution details"""
        try:
            with self.get_session() as session:
                signal = session.query(TradingSignal).filter(
                    TradingSignal.signal_id == signal_id
                ).first()
                
                if signal:
                    signal.executed = True
                    signal.execution_price = execution_price
                    signal.execution_timestamp = execution_timestamp
                    
                    logger.info("Signal execution updated", 
                               signal_id=signal_id,
                               execution_price=execution_price)
                    return True
                else:
                    logger.warning("Signal not found for execution update", 
                                 signal_id=signal_id)
                    return False
                    
        except Exception as e:
            logger.error("Failed to update signal execution", 
                        signal_id=signal_id, error=str(e))
            return False
    
    # === FEATURE DATA OPERATIONS ===
    
    def save_feature_data(
        self,
        symbol: str,
        timestamp: datetime,
        timeframe: str,
        features: Dict[str, Dict[str, float]],
        feature_version: str
    ) -> bool:
        """Save engineered features to database"""
        try:
            with self.get_session() as session:
                total_features = sum(len(cat_features) for cat_features in features.values())
                missing_features = sum(
                    1 for cat_features in features.values() 
                    for value in cat_features.values() 
                    if value is None or pd.isna(value)
                )
                
                feature_data = FeatureData(
                    symbol=symbol.upper(),
                    timestamp=timestamp,
                    timeframe=timeframe,
                    technical_features=features.get('technical', {}),
                    sentiment_features=features.get('sentiment', {}),
                    fundamental_features=features.get('fundamental', {}),
                    market_structure_features=features.get('market_structure', {}),
                    macro_features=features.get('macro', {}),
                    feature_version=feature_version,
                    total_features=total_features,
                    missing_features=missing_features,
                    quality_score=1.0 - (missing_features / max(total_features, 1))
                )
                
                session.merge(feature_data)
                
            logger.debug("Feature data saved", 
                        symbol=symbol, 
                        features=total_features,
                        version=feature_version)
            return True
            
        except Exception as e:
            logger.error("Failed to save feature data", 
                        symbol=symbol, error=str(e))
            return False
    
    # === PERFORMANCE METRICS OPERATIONS ===
    
    def save_performance_metrics(
        self,
        strategy_name: str,
        date: datetime,
        metrics: Dict[str, float]
    ) -> bool:
        """Save performance metrics for a strategy"""
        try:
            with self.get_session() as session:
                performance = PerformanceMetrics(
                    strategy_name=strategy_name,
                    date=date,
                    **metrics
                )
                
                session.merge(performance)
                
            logger.debug("Performance metrics saved", 
                        strategy=strategy_name,
                        date=date)
            return True
            
        except Exception as e:
            logger.error("Failed to save performance metrics", 
                        strategy=strategy_name, error=str(e))
            return False
    
    # === UTILITY OPERATIONS ===
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with self.get_session() as session:
                stats = {}
                
                # Table row counts
                tables = [
                    MarketData, SentimentData, FundamentalData,
                    TradingSignal, FeatureData, PerformanceMetrics
                ]
                
                for table in tables:
                    count = session.query(func.count(table.id)).scalar()
                    stats[table.__tablename__] = count
                
                # Date ranges
                market_data_range = session.query(
                    func.min(MarketData.timestamp),
                    func.max(MarketData.timestamp)
                ).first()
                
                if market_data_range[0]:
                    stats['market_data_date_range'] = {
                        'start': market_data_range[0].isoformat(),
                        'end': market_data_range[1].isoformat()
                    }
                
                # Database size (SQLite only)
                if self.db_type == 'sqlite':
                    db_path = self.database_url.replace('sqlite:///', '')
                    if os.path.exists(db_path):
                        stats['database_size_mb'] = os.path.getsize(db_path) / (1024 * 1024)
                
                logger.debug("Database stats retrieved", tables=len(stats))
                return stats
                
        except Exception as e:
            logger.error("Failed to get database stats", error=str(e))
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> Dict[str, int]:
        """Clean up old data based on retention policy"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            deleted_counts = {}
            
            with self.get_session() as session:
                
                # Clean old market data (keep daily data longer)
                deleted_market = session.query(MarketData).filter(
                    and_(
                        MarketData.timestamp < cutoff_date,
                        MarketData.timeframe.in_(['1min', '5min', '15min'])
                    )
                ).delete()
                deleted_counts['market_data'] = deleted_market
                
                # Clean old sentiment data
                deleted_sentiment = session.query(SentimentData).filter(
                    SentimentData.timestamp < cutoff_date
                ).delete()
                deleted_counts['sentiment_data'] = deleted_sentiment
                
                # Clean old executed signals
                deleted_signals = session.query(TradingSignal).filter(
                    and_(
                        TradingSignal.timestamp < cutoff_date,
                        TradingSignal.executed == True
                    )
                ).delete()
                deleted_counts['trading_signals'] = deleted_signals
                
            logger.info("Old data cleaned up", 
                       cutoff_date=cutoff_date.isoformat(),
                       deleted_counts=deleted_counts)
            
            return deleted_counts
            
        except Exception as e:
            logger.error("Failed to cleanup old data", error=str(e))
            return {}
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_session() as session:
                session.execute(text('SELECT 1'))
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error("Database connection test failed", error=str(e))
            return False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup database connections"""
        if hasattr(self, 'engine'):
            self.engine.dispose()
        logger.info("Database manager closed")
    
    def __str__(self) -> str:
        return f"DatabaseManager(type={self.db_type})"
    
    def __repr__(self) -> str:
        return self.__str__()