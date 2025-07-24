"""
Base Data Fetcher for QuantumSentiment Trading Bot

Provides high-level data fetching orchestration with:
- Scheduled data collection
- Multi-source data aggregation
- Real-time and batch processing
- Data validation and error handling
- Database persistence
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import structlog
from enum import Enum

from .data_interface import DataInterface, DataConfig

logger = structlog.get_logger(__name__)


class DataFrequency(Enum):
    """Data collection frequencies"""
    REALTIME = "realtime"
    MINUTE = "1min"
    FIVE_MINUTE = "5min"
    FIFTEEN_MINUTE = "15min"
    HOURLY = "1h"
    DAILY = "1d"
    WEEKLY = "1w"


class DataType(Enum):
    """Types of data to collect"""
    MARKET_DATA = "market_data"
    SENTIMENT_DATA = "sentiment_data"
    FUNDAMENTAL_DATA = "fundamental_data"
    NEWS_DATA = "news_data"
    ECONOMIC_DATA = "economic_data"


@dataclass
class DataSubscription:
    """Configuration for data collection subscription"""
    symbols: List[str]
    data_type: DataType
    frequency: DataFrequency
    callback: Optional[Callable] = None
    enabled: bool = True
    last_updated: Optional[datetime] = None
    error_count: int = 0
    max_errors: int = 5


@dataclass
class FetcherConfig:
    """Configuration for data fetcher"""
    # Database settings
    database_url: str = "sqlite:///data/quantum.db"
    
    # Collection settings
    max_workers: int = 10
    batch_size: int = 50
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Data retention
    max_history_days: int = 365
    cleanup_frequency: DataFrequency = DataFrequency.DAILY
    
    # Default subscriptions
    default_symbols: List[str] = field(default_factory=lambda: [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'SPY', 'QQQ',
        'BTC', 'ETH', 'BNB', 'ADA', 'SOL'
    ])


class DataFetcher:
    """High-level data fetching orchestrator"""
    
    def __init__(
        self,
        config: Optional[FetcherConfig] = None,
        data_interface: Optional[DataInterface] = None
    ):
        """
        Initialize data fetcher
        
        Args:
            config: Fetcher configuration
            data_interface: Data interface instance
        """
        self.config = config or FetcherConfig()
        self.data_interface = data_interface or DataInterface()
        
        # Subscriptions management
        self.subscriptions: Dict[str, DataSubscription] = {}
        self.running = False
        self.scheduler_task = None
        
        # Thread pool for concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Statistics
        self.stats = {
            'total_fetches': 0,
            'successful_fetches': 0,
            'failed_fetches': 0,
            'last_fetch_time': None,
            'start_time': datetime.now()
        }
        
        logger.info("Data fetcher initialized")
    
    # === SUBSCRIPTION MANAGEMENT ===
    
    def add_subscription(
        self,
        name: str,
        symbols: List[str],
        data_type: DataType,
        frequency: DataFrequency,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Add a data subscription
        
        Args:
            name: Unique subscription name
            symbols: List of symbols to track
            data_type: Type of data to collect
            frequency: Collection frequency
            callback: Optional callback function for data
            
        Returns:
            Subscription ID
        """
        subscription = DataSubscription(
            symbols=symbols,
            data_type=data_type,
            frequency=frequency,
            callback=callback
        )
        
        self.subscriptions[name] = subscription
        
        logger.info("Subscription added",
                   name=name,
                   symbols=len(symbols),
                   data_type=data_type.value,
                   frequency=frequency.value)
        
        return name
    
    def remove_subscription(self, name: str) -> bool:
        """Remove a data subscription"""
        if name in self.subscriptions:
            del self.subscriptions[name]
            logger.info("Subscription removed", name=name)
            return True
        return False
    
    def get_subscription(self, name: str) -> Optional[DataSubscription]:
        """Get subscription by name"""
        return self.subscriptions.get(name)
    
    def list_subscriptions(self) -> Dict[str, DataSubscription]:
        """List all subscriptions"""
        return self.subscriptions.copy()
    
    # === DATA FETCHING ===
    
    async def fetch_market_data(
        self,
        symbols: List[str],
        timeframe: str = '1H',
        days_back: int = 1
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch market data for multiple symbols
        
        Args:
            symbols: List of symbols
            timeframe: Data timeframe
            days_back: Days of historical data
            
        Returns:
            Dictionary of symbol -> DataFrame
        """
        results = {}
        
        def fetch_symbol_data(symbol: str) -> tuple:
            try:
                df = self.data_interface.get_historical_prices(
                    symbol=symbol,
                    timeframe=timeframe,
                    days_back=days_back
                )
                return symbol, df
            except Exception as e:
                logger.error("Failed to fetch market data", 
                           symbol=symbol, error=str(e))
                return symbol, pd.DataFrame()
        
        # Fetch data concurrently
        with ThreadPoolExecutor(max_workers=min(len(symbols), self.config.max_workers)) as executor:
            future_to_symbol = {
                executor.submit(fetch_symbol_data, symbol): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol, df = future.result()
                results[symbol] = df
                
                if not df.empty:
                    self.stats['successful_fetches'] += 1
                else:
                    self.stats['failed_fetches'] += 1
                
                self.stats['total_fetches'] += 1
        
        logger.info("Market data fetched",
                   symbols=len(symbols),
                   successful=len([r for r in results.values() if not r.empty]))
        
        return results
    
    async def fetch_sentiment_data(
        self,
        symbols: List[str],
        sources: Optional[List[str]] = None,
        hours_back: int = 24
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch sentiment data for multiple symbols
        
        Args:
            symbols: List of symbols
            sources: Sentiment sources to use
            hours_back: Hours of historical sentiment
            
        Returns:
            Dictionary of symbol -> sentiment data
        """
        results = {}
        
        def fetch_symbol_sentiment(symbol: str) -> tuple:
            try:
                sentiment = self.data_interface.get_sentiment_analysis(
                    symbol=symbol,
                    sources=sources,
                    hours_back=hours_back
                )
                return symbol, sentiment
            except Exception as e:
                logger.error("Failed to fetch sentiment data",
                           symbol=symbol, error=str(e))
                return symbol, {}
        
        # Fetch sentiment concurrently
        with ThreadPoolExecutor(max_workers=min(len(symbols), self.config.max_workers)) as executor:
            future_to_symbol = {
                executor.submit(fetch_symbol_sentiment, symbol): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol, sentiment = future.result()
                results[symbol] = sentiment
                
                if sentiment.get('confidence', 0) > 0:
                    self.stats['successful_fetches'] += 1
                else:
                    self.stats['failed_fetches'] += 1
                
                self.stats['total_fetches'] += 1
        
        logger.info("Sentiment data fetched",
                   symbols=len(symbols),
                   successful=len([r for r in results.values() if r.get('confidence', 0) > 0]))
        
        return results
    
    async def fetch_fundamental_data(
        self,
        symbols: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch fundamental data for multiple symbols"""
        results = {}
        
        def fetch_symbol_fundamentals(symbol: str) -> tuple:
            try:
                fundamentals = self.data_interface.get_company_fundamentals(symbol)
                return symbol, fundamentals
            except Exception as e:
                logger.error("Failed to fetch fundamental data",
                           symbol=symbol, error=str(e))
                return symbol, {}
        
        # Fetch fundamentals with rate limiting (slower than market data)
        for symbol in symbols:
            try:
                symbol_data, fundamentals = fetch_symbol_fundamentals(symbol)
                results[symbol_data] = fundamentals
                
                if fundamentals:
                    self.stats['successful_fetches'] += 1
                else:
                    self.stats['failed_fetches'] += 1
                
                self.stats['total_fetches'] += 1
                
                # Rate limiting for fundamental data
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error("Failed to fetch fundamentals", symbol=symbol, error=str(e))
                results[symbol] = {}
        
        logger.info("Fundamental data fetched",
                   symbols=len(symbols),
                   successful=len([r for r in results.values() if r]))
        
        return results
    
    async def fetch_quotes(
        self,
        symbols: List[str],
        asset_type: str = 'auto'
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch current quotes for multiple symbols"""
        try:
            quotes = self.data_interface.get_multiple_quotes(symbols, asset_type)
            
            successful = len([q for q in quotes.values() if q.get('price', 0) > 0])
            self.stats['successful_fetches'] += successful
            self.stats['failed_fetches'] += len(symbols) - successful
            self.stats['total_fetches'] += len(symbols)
            
            logger.debug("Quotes fetched",
                        symbols=len(symbols),
                        successful=successful)
            
            return quotes
            
        except Exception as e:
            logger.error("Failed to fetch quotes", error=str(e))
            return {symbol: {} for symbol in symbols}
    
    # === BATCH PROCESSING ===
    
    async def process_subscription(self, name: str, subscription: DataSubscription):
        """Process a single subscription"""
        try:
            if not subscription.enabled:
                return
            
            # Check if it's time to update based on frequency
            if subscription.last_updated:
                time_diff = datetime.now() - subscription.last_updated
                
                frequency_intervals = {
                    DataFrequency.MINUTE: timedelta(minutes=1),
                    DataFrequency.FIVE_MINUTE: timedelta(minutes=5),
                    DataFrequency.FIFTEEN_MINUTE: timedelta(minutes=15),
                    DataFrequency.HOURLY: timedelta(hours=1),
                    DataFrequency.DAILY: timedelta(days=1),
                    DataFrequency.WEEKLY: timedelta(weeks=1)
                }
                
                required_interval = frequency_intervals.get(subscription.frequency, timedelta(hours=1))
                
                if time_diff < required_interval:
                    return  # Too early to update
            
            # Fetch data based on type
            data = None
            
            if subscription.data_type == DataType.MARKET_DATA:
                # Determine appropriate timeframe based on frequency
                timeframe_map = {
                    DataFrequency.MINUTE: '1Min',
                    DataFrequency.FIVE_MINUTE: '5Min',
                    DataFrequency.FIFTEEN_MINUTE: '15Min',
                    DataFrequency.HOURLY: '1H',
                    DataFrequency.DAILY: '1D'
                }
                timeframe = timeframe_map.get(subscription.frequency, '1H')
                
                data = await self.fetch_market_data(
                    symbols=subscription.symbols,
                    timeframe=timeframe,
                    days_back=1
                )
                
            elif subscription.data_type == DataType.SENTIMENT_DATA:
                hours_back = {
                    DataFrequency.HOURLY: 1,
                    DataFrequency.DAILY: 24,
                    DataFrequency.WEEKLY: 168
                }.get(subscription.frequency, 24)
                
                data = await self.fetch_sentiment_data(
                    symbols=subscription.symbols,
                    hours_back=hours_back
                )
                
            elif subscription.data_type == DataType.FUNDAMENTAL_DATA:
                data = await self.fetch_fundamental_data(
                    symbols=subscription.symbols
                )
            
            if data:
                # Call callback if provided
                if subscription.callback:
                    try:
                        await subscription.callback(data)
                    except Exception as e:
                        logger.error("Subscription callback failed",
                                   name=name, error=str(e))
                
                # Update subscription
                subscription.last_updated = datetime.now()
                subscription.error_count = 0
                
                logger.debug("Subscription processed successfully", name=name)
            else:
                subscription.error_count += 1
                logger.warning("Subscription processing failed", 
                             name=name, 
                             error_count=subscription.error_count)
                
                # Disable subscription if too many errors
                if subscription.error_count >= subscription.max_errors:
                    subscription.enabled = False
                    logger.error("Subscription disabled due to errors", name=name)
        
        except Exception as e:
            subscription.error_count += 1
            logger.error("Subscription processing error",
                       name=name, error=str(e))
    
    async def process_all_subscriptions(self):
        """Process all active subscriptions"""
        if not self.subscriptions:
            return
        
        tasks = []
        for name, subscription in self.subscriptions.items():
            if subscription.enabled:
                task = asyncio.create_task(
                    self.process_subscription(name, subscription)
                )
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            self.stats['last_fetch_time'] = datetime.now()
    
    # === SCHEDULER ===
    
    async def start_scheduler(self, interval_seconds: int = 60):
        """Start the data collection scheduler"""
        self.running = True
        
        logger.info("Data fetcher scheduler started", interval=interval_seconds)
        
        while self.running:
            try:
                start_time = time.time()
                
                await self.process_all_subscriptions()
                
                processing_time = time.time() - start_time
                logger.debug("Subscription processing completed",
                           processing_time=round(processing_time, 2))
                
                # Wait for next interval
                await asyncio.sleep(max(0, interval_seconds - processing_time))
                
            except Exception as e:
                logger.error("Scheduler error", error=str(e))
                await asyncio.sleep(interval_seconds)
    
    def stop_scheduler(self):
        """Stop the data collection scheduler"""
        self.running = False
        logger.info("Data fetcher scheduler stopped")
    
    # === UTILITY METHODS ===
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get fetcher statistics"""
        uptime = datetime.now() - self.stats['start_time']
        
        return {
            'total_fetches': self.stats['total_fetches'],
            'successful_fetches': self.stats['successful_fetches'],
            'failed_fetches': self.stats['failed_fetches'],
            'success_rate': (
                self.stats['successful_fetches'] / max(self.stats['total_fetches'], 1) * 100
            ),
            'uptime_seconds': uptime.total_seconds(),
            'last_fetch_time': self.stats['last_fetch_time'],
            'active_subscriptions': len([s for s in self.subscriptions.values() if s.enabled]),
            'total_subscriptions': len(self.subscriptions),
            'running': self.running
        }
    
    def create_default_subscriptions(self):
        """Create default data subscriptions"""
        
        # Market data - high frequency for active trading
        self.add_subscription(
            name="market_data_1min",
            symbols=self.config.default_symbols,
            data_type=DataType.MARKET_DATA,
            frequency=DataFrequency.MINUTE
        )
        
        # Sentiment data - moderate frequency
        self.add_subscription(
            name="sentiment_data_15min",
            symbols=self.config.default_symbols[:10],  # Limit for rate limiting
            data_type=DataType.SENTIMENT_DATA,
            frequency=DataFrequency.FIFTEEN_MINUTE
        )
        
        # Fundamental data - low frequency
        stock_symbols = [s for s in self.config.default_symbols if s not in ['BTC', 'ETH', 'BNB', 'ADA', 'SOL']]
        self.add_subscription(
            name="fundamental_data_daily",
            symbols=stock_symbols,
            data_type=DataType.FUNDAMENTAL_DATA,
            frequency=DataFrequency.DAILY
        )
        
        logger.info("Default subscriptions created", count=len(self.subscriptions))
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on data sources"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'data_interface': {},
            'subscriptions': {},
            'statistics': self.get_statistics()
        }
        
        try:
            # Check data interface status
            data_status = self.data_interface.get_data_status()
            health_status['data_interface'] = data_status
            
            # Check subscription health
            for name, subscription in self.subscriptions.items():
                health_status['subscriptions'][name] = {
                    'enabled': subscription.enabled,
                    'error_count': subscription.error_count,
                    'last_updated': subscription.last_updated.isoformat() if subscription.last_updated else None,
                    'symbols_count': len(subscription.symbols)
                }
            
            # Determine overall health
            failed_clients = len([
                c for c in data_status.get('clients', {}).values() 
                if c.get('status') != 'connected'
            ])
            
            disabled_subscriptions = len([
                s for s in self.subscriptions.values() 
                if not s.enabled
            ])
            
            if failed_clients > 0 or disabled_subscriptions > len(self.subscriptions) / 2:
                health_status['overall_status'] = 'degraded'
            
            if failed_clients >= len(data_status.get('clients', {})):
                health_status['overall_status'] = 'unhealthy'
        
        except Exception as e:
            health_status['overall_status'] = 'error'
            health_status['error'] = str(e)
            logger.error("Health check failed", error=str(e))
        
        return health_status
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources"""
        self.stop_scheduler()
        self.executor.shutdown(wait=True)
        logger.info("Data fetcher closed")
    
    def __str__(self) -> str:
        return f"DataFetcher(subscriptions={len(self.subscriptions)}, running={self.running})"
    
    def __repr__(self) -> str:
        return self.__str__()