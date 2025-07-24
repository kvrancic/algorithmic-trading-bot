"""
Unified Data Interface for QuantumSentiment Trading Bot

This module provides a unified interface to all data sources:
- Market data (stocks, crypto, forex)
- Sentiment data (Reddit, Twitter, news)
- Fundamental data (financials, economic indicators)
- Alternative data (political trades, social metrics)

The interface abstracts away the complexity of individual APIs
and provides a consistent way to access all data.
"""

import os
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import structlog
from dataclasses import dataclass, field

from .alpaca_client import AlpacaClient
from .reddit_client import RedditClient
from .alpha_vantage_client import AlphaVantageClient
from .crypto_client import CryptoClient

logger = structlog.get_logger(__name__)


@dataclass
class DataConfig:
    """Configuration for data sources"""
    # API credentials (will be loaded from environment)
    alpaca_paper: bool = True
    
    # Data source preferences
    preferred_crypto_source: str = 'coingecko'
    preferred_sentiment_sources: List[str] = field(default_factory=lambda: ['reddit'])
    
    # Rate limiting and caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes
    max_concurrent_requests: int = 10
    
    # Data filtering
    min_market_cap: float = 1e9  # $1B minimum market cap
    min_volume: float = 1e6      # $1M minimum daily volume
    max_symbols_per_request: int = 50


class DataInterface:
    """Unified data interface for all market and sentiment data"""
    
    def __init__(self, config: Optional[DataConfig] = None):
        """
        Initialize data interface with all clients
        
        Args:
            config: Data configuration options
        """
        self.config = config or DataConfig()
        
        # Initialize clients
        self.clients = {}
        self._init_clients()
        
        # Thread pool for concurrent requests
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests)
        
        # Simple in-memory cache
        self.cache = {}
        self.cache_timestamps = {}
        
        logger.info("Data interface initialized", 
                   clients=list(self.clients.keys()))
    
    def _init_clients(self):
        """Initialize all data clients"""
        try:
            # Alpaca client (always try to initialize)
            self.clients['alpaca'] = AlpacaClient(paper=self.config.alpaca_paper)
            logger.info("Alpaca client initialized")
        except Exception as e:
            logger.warning("Failed to initialize Alpaca client", error=str(e))
        
        try:
            # Reddit client
            self.clients['reddit'] = RedditClient()
            logger.info("Reddit client initialized")
        except Exception as e:
            logger.warning("Failed to initialize Reddit client", error=str(e))
        
        try:
            # Alpha Vantage client
            self.clients['alphavantage'] = AlphaVantageClient()
            logger.info("Alpha Vantage client initialized")
        except Exception as e:
            logger.warning("Failed to initialize Alpha Vantage client", error=str(e))
        
        try:
            # Crypto client
            self.clients['crypto'] = CryptoClient()
            logger.info("Crypto client initialized")
        except Exception as e:
            logger.warning("Failed to initialize Crypto client", error=str(e))
    
    # === CACHE MANAGEMENT ===
    
    def _get_cache_key(self, method: str, *args, **kwargs) -> str:
        """Generate cache key for method call"""
        key_parts = [method] + [str(arg) for arg in args]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        return "|".join(key_parts)
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if not self.config.enable_caching:
            return False
        
        if cache_key not in self.cache_timestamps:
            return False
        
        age = datetime.now().timestamp() - self.cache_timestamps[cache_key]
        return age < self.config.cache_ttl_seconds
    
    def _get_from_cache(self, cache_key: str) -> Any:
        """Get data from cache if valid"""
        if self._is_cache_valid(cache_key):
            logger.debug("Cache hit", key=cache_key)
            return self.cache[cache_key]
        return None
    
    def _set_cache(self, cache_key: str, data: Any):
        """Store data in cache"""
        if self.config.enable_caching:
            self.cache[cache_key] = data
            self.cache_timestamps[cache_key] = datetime.now().timestamp()
            logger.debug("Cache set", key=cache_key)
    
    # === MARKET DATA ===
    
    def get_quote(self, symbol: str, asset_type: str = 'auto') -> Dict[str, Any]:
        """
        Get current quote for any asset
        
        Args:
            symbol: Asset symbol (AAPL, BTC, etc.)
            asset_type: 'stock', 'crypto', or 'auto' to detect
            
        Returns:
            Unified quote data
        """
        cache_key = self._get_cache_key('get_quote', symbol, asset_type)
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        # Auto-detect asset type
        if asset_type == 'auto':
            if symbol.upper() in ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'MATIC', 'DOT', 'AVAX', 'LINK', 'DOGE']:
                asset_type = 'crypto'
            else:
                asset_type = 'stock'
        
        try:
            if asset_type == 'crypto' and 'crypto' in self.clients:
                quote = self.clients['crypto'].get_crypto_quote(
                    symbol, 
                    self.config.preferred_crypto_source
                )
                
            elif asset_type == 'stock':
                # Try Alpaca first, then Alpha Vantage
                if 'alpaca' in self.clients:
                    try:
                        alpaca_quote = self.clients['alpaca'].get_latest_trade(symbol)
                        quote = {
                            'symbol': symbol,
                            'price': alpaca_quote.get('price', 0),
                            'timestamp': alpaca_quote.get('timestamp'),
                            'source': 'alpaca'
                        }
                    except Exception:
                        if 'alphavantage' in self.clients:
                            av_quote = self.clients['alphavantage'].get_quote(symbol)
                            quote = {
                                'symbol': symbol,
                                'price': av_quote.get('price', 0),
                                'change': av_quote.get('change', 0),
                                'change_percent': av_quote.get('change_percent', 0),
                                'volume': av_quote.get('volume', 0),
                                'source': 'alphavantage'
                            }
                        else:
                            quote = {}
                else:
                    quote = {}
            else:
                quote = {}
            
            self._set_cache(cache_key, quote)
            
            logger.debug("Retrieved quote", 
                        symbol=symbol, 
                        asset_type=asset_type,
                        price=quote.get('price', 0))
            
            return quote
            
        except Exception as e:
            logger.error("Failed to get quote", 
                        symbol=symbol, 
                        asset_type=asset_type, 
                        error=str(e))
            return {}
    
    def get_historical_prices(
        self,
        symbol: str,
        timeframe: str = '1D',
        days_back: int = 30,
        asset_type: str = 'auto'
    ) -> pd.DataFrame:
        """
        Get historical price data for any asset
        
        Args:
            symbol: Asset symbol
            timeframe: Data frequency ('1Min', '1H', '1D')
            days_back: Number of days of history
            asset_type: 'stock', 'crypto', or 'auto'
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = self._get_cache_key('get_historical_prices', symbol, timeframe, days_back, asset_type)
        cached = self._get_from_cache(cache_key)
        if cached is not None and not cached.empty:
            return cached
        
        # Auto-detect asset type
        if asset_type == 'auto':
            if symbol.upper() in ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'MATIC', 'DOT', 'AVAX', 'LINK', 'DOGE']:
                asset_type = 'crypto'
            else:
                asset_type = 'stock'
        
        try:
            if asset_type == 'crypto' and 'crypto' in self.clients:
                # Map timeframe to crypto client format
                interval = 'daily' if timeframe == '1D' else 'hourly'
                df = self.clients['crypto'].get_crypto_historical(
                    symbol, 
                    days_back, 
                    interval,
                    self.config.preferred_crypto_source
                )
                
            elif asset_type == 'stock':
                if 'alpaca' in self.clients:
                    # Map timeframe to Alpaca format
                    alpaca_timeframe = {
                        '1Min': '1Min',
                        '5Min': '5Min', 
                        '15Min': '15Min',
                        '1H': '1Hour',
                        '1D': '1Day'
                    }.get(timeframe, '1Hour')
                    
                    start_date = datetime.now() - timedelta(days=days_back)
                    df = self.clients['alpaca'].get_bars(
                        symbol, 
                        alpaca_timeframe, 
                        start=start_date
                    )
                    
                elif 'alphavantage' in self.clients:
                    # Use Alpha Vantage as fallback
                    if timeframe == '1D':
                        df = self.clients['alphavantage'].get_daily_prices(symbol)
                    else:
                        df = self.clients['alphavantage'].get_intraday_prices(
                            symbol, 
                            f"{timeframe.lower().replace('min', 'min').replace('h', 'min')}"
                        )
                else:
                    df = pd.DataFrame()
            else:
                df = pd.DataFrame()
            
            if not df.empty:
                # Standardize column names
                if 'price' in df.columns and 'open' not in df.columns:
                    # Crypto data - create OHLC from price
                    df['open'] = df['price']
                    df['high'] = df['price'] 
                    df['low'] = df['price']
                    df['close'] = df['price']
                
                # Ensure we have standard OHLCV columns
                standard_cols = ['open', 'high', 'low', 'close', 'volume']
                df = df.reindex(columns=standard_cols, fill_value=0)
                
                self._set_cache(cache_key, df)
                
                logger.debug("Retrieved historical prices", 
                           symbol=symbol,
                           asset_type=asset_type,
                           timeframe=timeframe,
                           records=len(df))
            
            return df
            
        except Exception as e:
            logger.error("Failed to get historical prices", 
                        symbol=symbol, 
                        timeframe=timeframe,
                        error=str(e))
            return pd.DataFrame()
    
    def get_multiple_quotes(
        self,
        symbols: List[str],
        asset_type: str = 'auto'
    ) -> Dict[str, Dict[str, Any]]:
        """Get quotes for multiple symbols concurrently"""
        
        def get_single_quote(symbol):
            return symbol, self.get_quote(symbol, asset_type)
        
        results = {}
        
        # Use thread pool for concurrent requests
        with ThreadPoolExecutor(max_workers=min(len(symbols), self.config.max_concurrent_requests)) as executor:
            future_to_symbol = {
                executor.submit(get_single_quote, symbol): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                try:
                    symbol, quote = future.result()
                    results[symbol] = quote
                except Exception as e:
                    symbol = future_to_symbol[future]
                    logger.error("Failed to get quote in batch", symbol=symbol, error=str(e))
                    results[symbol] = {}
        
        logger.info("Retrieved multiple quotes", 
                   requested=len(symbols),
                   successful=len([q for q in results.values() if q]))
        
        return results
    
    # === SENTIMENT DATA ===
    
    def get_sentiment_analysis(
        self,
        symbol: str,
        sources: Optional[List[str]] = None,
        hours_back: int = 24
    ) -> Dict[str, Any]:
        """
        Get sentiment analysis for symbol from multiple sources
        
        Args:
            symbol: Asset symbol
            sources: List of sentiment sources to use
            hours_back: Hours of historical sentiment
            
        Returns:
            Aggregated sentiment data
        """
        cache_key = self._get_cache_key('get_sentiment_analysis', symbol, str(sources), hours_back)
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        sources = sources or self.config.preferred_sentiment_sources
        sentiment_data = {
            'symbol': symbol,
            'overall_sentiment': 0.0,
            'confidence': 0.0,
            'sources': {},
            'aggregated_at': datetime.now().isoformat()
        }
        
        # Collect sentiment from each source
        total_weight = 0
        weighted_sentiment = 0
        
        if 'reddit' in sources and 'reddit' in self.clients:
            try:
                reddit_sentiment = self.clients['reddit'].analyze_ticker_sentiment(
                    symbol, 
                    hours_back=hours_back
                )
                
                if reddit_sentiment.get('confidence', 0) > 0.1:
                    weight = reddit_sentiment.get('confidence', 0) * reddit_sentiment.get('mention_count', 1)
                    weighted_sentiment += reddit_sentiment.get('sentiment_score', 0) * weight
                    total_weight += weight
                    
                    sentiment_data['sources']['reddit'] = reddit_sentiment
                    
            except Exception as e:
                logger.error("Failed to get Reddit sentiment", symbol=symbol, error=str(e))
        
        # TODO: Add Twitter, news, and other sentiment sources when implemented
        
        # Calculate final sentiment
        if total_weight > 0:
            sentiment_data['overall_sentiment'] = weighted_sentiment / total_weight
            sentiment_data['confidence'] = min(total_weight / 100, 1.0)  # Normalize confidence
        
        self._set_cache(cache_key, sentiment_data)
        
        logger.debug("Retrieved sentiment analysis", 
                    symbol=symbol,
                    sentiment=sentiment_data['overall_sentiment'],
                    confidence=sentiment_data['confidence'])
        
        return sentiment_data
    
    def get_trending_topics(
        self,
        source: str = 'reddit',
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get trending topics/tickers from sentiment sources"""
        
        cache_key = self._get_cache_key('get_trending_topics', source, limit)
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            if source == 'reddit' and 'reddit' in self.clients:
                trending = self.clients['reddit'].get_trending_tickers(
                    limit=limit*2  # Get more than needed for filtering
                )
                
                # Filter and rank
                filtered_trending = []
                for item in trending[:limit]:
                    if item.get('mentions', 0) >= 3:  # Minimum mentions
                        filtered_trending.append(item)
                
                self._set_cache(cache_key, filtered_trending)
                
                logger.debug("Retrieved trending topics", 
                           source=source, 
                           count=len(filtered_trending))
                
                return filtered_trending
            
        except Exception as e:
            logger.error("Failed to get trending topics", source=source, error=str(e))
        
        return []
    
    # === FUNDAMENTAL DATA ===
    
    def get_company_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Get company fundamental data"""
        
        cache_key = self._get_cache_key('get_company_fundamentals', symbol)
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            if 'alphavantage' in self.clients:
                fundamentals = self.clients['alphavantage'].get_company_overview(symbol)
                
                self._set_cache(cache_key, fundamentals)
                
                logger.debug("Retrieved company fundamentals", 
                           symbol=symbol,
                           pe_ratio=fundamentals.get('PERatio'))
                
                return fundamentals
            
        except Exception as e:
            logger.error("Failed to get company fundamentals", symbol=symbol, error=str(e))
        
        return {}
    
    # === PORTFOLIO & ACCOUNT DATA ===
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get trading account information"""
        try:
            if 'alpaca' in self.clients:
                account = self.clients['alpaca'].get_account()
                
                return {
                    'account_id': account.id,
                    'status': account.status,
                    'buying_power': float(account.buying_power),
                    'portfolio_value': float(account.portfolio_value),
                    'equity': float(account.equity),
                    'cash': float(account.cash),
                    'day_trade_count': account.day_trade_count,
                    'pattern_day_trader': account.pattern_day_trader,
                    'trading_blocked': account.trading_blocked,
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error("Failed to get account info", error=str(e))
        
        return {}
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current portfolio positions"""
        try:
            if 'alpaca' in self.clients:
                positions = self.clients['alpaca'].get_positions()
                
                position_data = []
                for pos in positions:
                    position_data.append({
                        'symbol': pos.symbol,
                        'qty': float(pos.qty),
                        'side': pos.side,
                        'market_value': float(pos.market_value),
                        'cost_basis': float(pos.cost_basis),
                        'unrealized_pl': float(pos.unrealized_pl),
                        'unrealized_plpc': float(pos.unrealized_plpc),
                        'avg_entry_price': float(pos.avg_entry_price),
                        'current_price': float(pos.current_price)
                    })
                
                logger.debug("Retrieved positions", count=len(position_data))
                return position_data
                
        except Exception as e:
            logger.error("Failed to get positions", error=str(e))
        
        return []
    
    # === UTILITY METHODS ===
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            if 'alpaca' in self.clients:
                return self.clients['alpaca'].is_market_open()
        except Exception as e:
            logger.error("Failed to check market status", error=str(e))
        return False
    
    def get_data_status(self) -> Dict[str, Any]:
        """Get status of all data sources"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'clients': {},
            'cache': {
                'enabled': self.config.enable_caching,
                'entries': len(self.cache),
                'ttl_seconds': self.config.cache_ttl_seconds
            }
        }
        
        for name, client in self.clients.items():
            try:
                if name == 'alpaca':
                    account = client.get_account()
                    status['clients'][name] = {
                        'status': 'connected',
                        'account_status': account.status,
                        'paper_trading': client.paper_trading
                    }
                else:
                    status['clients'][name] = {'status': 'connected'}
                    
            except Exception as e:
                status['clients'][name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return status
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        self.cache_timestamps.clear()
        logger.info("Cache cleared")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        logger.info("Data interface closed")
    
    def __str__(self) -> str:
        return f"DataInterface(clients={len(self.clients)}, cache_entries={len(self.cache)})"
    
    def __repr__(self) -> str:
        return self.__str__()