"""
Data Fetcher

Unified interface for fetching data from multiple sources
including market data, sentiment data, and economic indicators.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import asyncio
import pandas as pd
import structlog

from .alpaca_client import AlpacaClient

logger = structlog.get_logger(__name__)


@dataclass
class FetcherConfig:
    """Configuration for data fetcher"""
    
    # Market data sources
    enable_alpaca: bool = True
    enable_alpha_vantage: bool = True
    
    # Data parameters
    max_symbols: int = 100
    default_timeframe: str = "1Day"
    lookback_days: int = 365
    
    # Rate limiting
    alpaca_rate_limit: int = 200  # requests per minute
    alpha_vantage_rate_limit: int = 5  # requests per minute
    
    # Data quality
    min_data_points: int = 10
    max_missing_ratio: float = 0.1
    
    # Storage settings
    cache_data: bool = True
    save_to_database: bool = True
    
    # Update frequencies (in minutes)
    market_data_update_freq: int = 1
    sentiment_data_update_freq: int = 5
    fundamental_data_update_freq: int = 60
    
    # Concurrency settings
    max_workers: int = 4
    enable_parallel_downloads: bool = True


class DataFetcher:
    """Unified data fetching interface"""
    
    def __init__(self, config, db_manager):
        """
        Initialize data fetcher
        
        Args:
            config: Configuration object
            db_manager: Database manager instance
        """
        self.config = config
        self.db_manager = db_manager
        
        # Initialize data clients
        self.alpaca_client = AlpacaClient()
        
        logger.info("Data fetcher initialized")
    
    async def get_market_data(
        self,
        symbols: List[str],
        timeframe: str = "1Hour",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch market data for multiple symbols
        
        Args:
            symbols: List of symbols to fetch
            timeframe: Data timeframe
            start: Start date
            end: End date
            limit: Maximum number of bars
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        for symbol in symbols:
            try:
                data = await self.alpaca_client.get_bars(
                    symbol=symbol,
                    timeframe=timeframe,
                    start=start,
                    end=end,
                    limit=limit
                )
                results[symbol] = data
                
            except Exception as e:
                logger.error("Failed to fetch market data",
                           symbol=symbol,
                           error=str(e))
                results[symbol] = pd.DataFrame()
        
        return results
    
    async def get_latest_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get latest quotes for symbols
        
        Args:
            symbols: List of symbols
            
        Returns:
            Dictionary mapping symbols to quote data
        """
        results = {}
        
        for symbol in symbols:
            try:
                quote = await self.alpaca_client.get_latest_quote(symbol)
                results[symbol] = quote
                
            except Exception as e:
                logger.error("Failed to fetch quote",
                           symbol=symbol,
                           error=str(e))
                results[symbol] = None
        
        return results
    
    async def get_sentiment_data(
        self,
        symbols: List[str],
        sources: Optional[List[str]] = None,
        lookback_hours: int = 24
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch sentiment data for symbols
        
        Args:
            symbols: List of symbols
            sources: List of sentiment sources
            lookback_hours: Hours to look back
            
        Returns:
            Dictionary mapping symbols to sentiment data
        """
        # Placeholder implementation
        # In a full implementation, this would fetch from Reddit, news, etc.
        
        results = {}
        for symbol in symbols:
            results[symbol] = {
                'sentiment_score': 0.0,
                'confidence': 0.5,
                'volume': 0,
                'sources': []
            }
        
        return results
    
    async def store_market_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        timeframe: str
    ) -> bool:
        """
        Store market data in database
        
        Args:
            symbol: Trading symbol
            data: Market data DataFrame
            timeframe: Data timeframe
            
        Returns:
            True if successful
        """
        try:
            # Store in database
            # This would use the database manager to store the data
            logger.debug("Market data stored",
                        symbol=symbol,
                        rows=len(data),
                        timeframe=timeframe)
            return True
            
        except Exception as e:
            logger.error("Failed to store market data",
                        symbol=symbol,
                        error=str(e))
            return False
    
    async def get_cached_data(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Get cached market data from database
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start: Start date
            end: End date
            
        Returns:
            Cached data or None if not available
        """
        try:
            # This would query the database for cached data
            # For now, return None to force fresh fetching
            return None
            
        except Exception as e:
            logger.error("Failed to get cached data",
                        symbol=symbol,
                        error=str(e))
            return None
    
    async def update_all_data(self, symbols: List[str]) -> None:
        """
        Update all data for given symbols
        
        Args:
            symbols: List of symbols to update
        """
        tasks = []
        
        # Fetch market data
        tasks.append(self.get_market_data(symbols))
        
        # Fetch sentiment data
        tasks.append(self.get_sentiment_data(symbols))
        
        # Execute all tasks concurrently
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            market_data, sentiment_data = results
            
            logger.info("Data update completed",
                       symbols=symbols,
                       market_data_count=len(market_data) if isinstance(market_data, dict) else 0,
                       sentiment_data_count=len(sentiment_data) if isinstance(sentiment_data, dict) else 0)
            
        except Exception as e:
            logger.error("Data update failed", error=str(e))
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        return self.alpaca_client.is_market_open()
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all data sources
        
        Returns:
            Dictionary mapping sources to health status
        """
        health = {}
        
        # Check Alpaca connection
        try:
            account = self.alpaca_client.get_account()
            health['alpaca'] = account is not None
        except:
            health['alpaca'] = False
        
        # Add other health checks here
        
        return health
    
    async def fetch_market_data(self, symbols: List[str], timeframe: str = "1Day", days_back: int = 365) -> Dict[str, pd.DataFrame]:
        """Fetch market data for symbols"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            results = {}
            for symbol in symbols:
                try:
                    data = self.alpaca_client.get_bars(
                        symbol=symbol,
                        timeframe=timeframe,
                        start=start_date,
                        end=end_date,
                        limit=1000
                    )
                    results[symbol] = data
                except Exception as e:
                    logger.error("Failed to fetch market data for symbol", symbol=symbol, error=str(e))
                    results[symbol] = pd.DataFrame()
            
            return results
        except Exception as e:
            logger.error("Failed to fetch market data", error=str(e))
            return {symbol: pd.DataFrame() for symbol in symbols}
    
    async def fetch_sentiment_data(self, symbols: List[str], sources: Optional[List[str]] = None, hours_back: int = 24) -> Dict[str, Dict[str, Any]]:
        """Fetch sentiment data for symbols"""
        try:
            # This would use sentiment analyzers
            # For now, return structured data matching what the script expects
            results = {}
            for symbol in symbols:
                results[symbol] = {
                    'data': [],  # Script expects 'data' field containing list of sentiment records
                    'sentiment_score': 0.0,
                    'confidence': 0.0,
                    'sources': sources or []
                }
            return results
        except Exception as e:
            logger.error("Failed to fetch sentiment data", error=str(e))
            return {symbol: {} for symbol in symbols}
    
    async def fetch_fundamental_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch fundamental data for symbols"""
        try:
            # This would use fundamental data sources
            # For now, return empty data for each symbol
            results = {}
            for symbol in symbols:
                results[symbol] = {
                    'symbol': symbol,
                    'pe_ratio': None,
                    'market_cap': None,
                    'revenue': None
                }
            return results
        except Exception as e:
            logger.error("Failed to fetch fundamental data", error=str(e))
            return {symbol: {} for symbol in symbols}