"""
Alpha Vantage API Client for QuantumSentiment Trading Bot

Handles interactions with Alpha Vantage API for:
- Fundamental data (earnings, financials)
- Economic indicators
- Technical indicators
- Market news and sentiment
"""

import os
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import structlog
import requests

logger = structlog.get_logger(__name__)

try:
    from alpha_vantage.timeseries import TimeSeries
    from alpha_vantage.fundamentaldata import FundamentalData
    from alpha_vantage.techindicators import TechIndicators
    from alpha_vantage.cryptocurrencies import CryptoCurrencies
    # SectorPerformances may not be available in newer versions
    try:
        from alpha_vantage.sectorperformance import SectorPerformances
    except ImportError:
        SectorPerformances = None
except ImportError as e:
    logger.warning("Alpha Vantage library imports failed", error=str(e))
    TimeSeries = None
    FundamentalData = None
    TechIndicators = None
    SectorPerformances = None
    CryptoCurrencies = None

logger = structlog.get_logger(__name__)


class AlphaVantageClient:
    """Alpha Vantage API client for fundamental and economic data"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Alpha Vantage client
        
        Args:
            api_key: Alpha Vantage API key (defaults to env var)
        """
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not provided")
        
        # Initialize Alpha Vantage modules
        self.timeseries = TimeSeries(key=self.api_key, output_format='pandas')
        self.fundamentals = FundamentalData(key=self.api_key, output_format='pandas')
        self.indicators = TechIndicators(key=self.api_key, output_format='pandas')
        self.sectors = SectorPerformances(key=self.api_key, output_format='pandas') if SectorPerformances else None
        self.crypto = CryptoCurrencies(key=self.api_key, output_format='pandas')
        
        # Rate limiting (5 calls per minute for free tier)
        self.last_call_time = 0
        self.min_interval = 12  # seconds between calls
        
        logger.info("Alpha Vantage client initialized")
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            logger.debug("Rate limiting", sleep_time=sleep_time)
            time.sleep(sleep_time)
        self.last_call_time = time.time()
    
    # === MARKET DATA ===
    
    def get_daily_prices(
        self,
        symbol: str,
        outputsize: str = 'compact'
    ) -> pd.DataFrame:
        """
        Get daily price data
        
        Args:
            symbol: Stock symbol
            outputsize: 'compact' (100 days) or 'full' (20+ years)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            self._rate_limit()
            data, meta_data = self.timeseries.get_daily(
                symbol=symbol,
                outputsize=outputsize
            )
            
            # Rename columns to standard format
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            data.index.name = 'date'
            data = data.sort_index()
            
            logger.debug("Retrieved daily prices", 
                        symbol=symbol, 
                        records=len(data))
            return data
            
        except Exception as e:
            logger.error("Failed to get daily prices", 
                        symbol=symbol, error=str(e))
            return pd.DataFrame()
    
    def get_intraday_prices(
        self,
        symbol: str,
        interval: str = '60min',
        outputsize: str = 'compact'
    ) -> pd.DataFrame:
        """Get intraday price data"""
        try:
            self._rate_limit()
            data, meta_data = self.timeseries.get_intraday(
                symbol=symbol,
                interval=interval,
                outputsize=outputsize
            )
            
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            data.index.name = 'timestamp'
            data = data.sort_index()
            
            logger.debug("Retrieved intraday prices", 
                        symbol=symbol, 
                        interval=interval,
                        records=len(data))
            return data
            
        except Exception as e:
            logger.error("Failed to get intraday prices", 
                        symbol=symbol, error=str(e))
            return pd.DataFrame()
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote"""
        try:
            self._rate_limit()
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'Global Quote' in data:
                quote_data = data['Global Quote']
                return {
                    'symbol': quote_data.get('01. symbol', symbol),
                    'price': float(quote_data.get('05. price', 0)),
                    'change': float(quote_data.get('09. change', 0)),
                    'change_percent': quote_data.get('10. change percent', '0%').strip('%'),
                    'volume': int(quote_data.get('06. volume', 0)),
                    'latest_trading_day': quote_data.get('07. latest trading day'),
                    'previous_close': float(quote_data.get('08. previous close', 0)),
                    'open': float(quote_data.get('02. open', 0)),
                    'high': float(quote_data.get('03. high', 0)),
                    'low': float(quote_data.get('04. low', 0))
                }
            else:
                logger.warning("No quote data returned", symbol=symbol, response=data)
                return {}
                
        except Exception as e:
            logger.error("Failed to get quote", symbol=symbol, error=str(e))
            return {}
    
    # === FUNDAMENTAL DATA ===
    
    def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """Get company fundamental data"""
        try:
            self._rate_limit()
            data = self.fundamentals.get_company_overview(symbol)[0]
            
            # Convert to dictionary and clean data
            overview = data.to_dict('records')[0] if not data.empty else {}
            
            # Convert numeric fields
            numeric_fields = [
                'MarketCapitalization', 'EBITDA', 'PERatio', 'PEGRatio',
                'BookValue', 'DividendPerShare', 'DividendYield', 'EPS',
                'RevenuePerShareTTM', 'ProfitMargin', 'OperatingMarginTTM',
                'ReturnOnAssetsTTM', 'ReturnOnEquityTTM', 'RevenueTTM',
                'GrossProfitTTM', 'DilutedEPSTTM', 'QuarterlyEarningsGrowthYOY',
                'QuarterlyRevenueGrowthYOY', 'AnalystTargetPrice', 'Beta',
                '52WeekHigh', '52WeekLow', '50DayMovingAverage', '200DayMovingAverage'
            ]
            
            for field in numeric_fields:
                if field in overview and overview[field] not in ['None', '-', '']:
                    try:
                        overview[field] = float(overview[field])
                    except (ValueError, TypeError):
                        overview[field] = None
            
            logger.debug("Retrieved company overview", symbol=symbol)
            return overview
            
        except Exception as e:
            logger.error("Failed to get company overview", symbol=symbol, error=str(e))
            return {}
    
    def get_income_statement(self, symbol: str) -> pd.DataFrame:
        """Get annual income statement"""
        try:
            self._rate_limit()
            data = self.fundamentals.get_income_statement_annual(symbol)[0]
            
            logger.debug("Retrieved income statement", symbol=symbol, records=len(data))
            return data
            
        except Exception as e:
            logger.error("Failed to get income statement", symbol=symbol, error=str(e))
            return pd.DataFrame()
    
    def get_balance_sheet(self, symbol: str) -> pd.DataFrame:
        """Get annual balance sheet"""
        try:
            self._rate_limit()
            data = self.fundamentals.get_balance_sheet_annual(symbol)[0]
            
            logger.debug("Retrieved balance sheet", symbol=symbol, records=len(data))
            return data
            
        except Exception as e:
            logger.error("Failed to get balance sheet", symbol=symbol, error=str(e))
            return pd.DataFrame()
    
    def get_cash_flow(self, symbol: str) -> pd.DataFrame:
        """Get annual cash flow statement"""
        try:
            self._rate_limit()
            data = self.fundamentals.get_cash_flow_annual(symbol)[0]
            
            logger.debug("Retrieved cash flow", symbol=symbol, records=len(data))
            return data
            
        except Exception as e:
            logger.error("Failed to get cash flow", symbol=symbol, error=str(e))
            return pd.DataFrame()
    
    def get_earnings(self, symbol: str) -> Dict[str, Any]:
        """Get earnings data"""
        try:
            self._rate_limit()
            annual_data = self.fundamentals.get_earnings(symbol)[0]
            quarterly_data = self.fundamentals.get_earnings(symbol)[1]
            
            return {
                'annual': annual_data,
                'quarterly': quarterly_data
            }
            
        except Exception as e:
            logger.error("Failed to get earnings", symbol=symbol, error=str(e))
            return {'annual': pd.DataFrame(), 'quarterly': pd.DataFrame()}
    
    # === TECHNICAL INDICATORS ===
    
    def get_sma(
        self,
        symbol: str,
        interval: str = 'daily',
        time_period: int = 20,
        series_type: str = 'close'
    ) -> pd.DataFrame:
        """Get Simple Moving Average"""
        try:
            self._rate_limit()
            data, meta_data = self.indicators.get_sma(
                symbol=symbol,
                interval=interval,
                time_period=time_period,
                series_type=series_type
            )
            
            logger.debug("Retrieved SMA", 
                        symbol=symbol, 
                        period=time_period,
                        records=len(data))
            return data
            
        except Exception as e:
            logger.error("Failed to get SMA", symbol=symbol, error=str(e))
            return pd.DataFrame()
    
    def get_rsi(
        self,
        symbol: str,
        interval: str = 'daily',
        time_period: int = 14,
        series_type: str = 'close'
    ) -> pd.DataFrame:
        """Get Relative Strength Index"""
        try:
            self._rate_limit()
            data, meta_data = self.indicators.get_rsi(
                symbol=symbol,
                interval=interval,
                time_period=time_period,
                series_type=series_type
            )
            
            logger.debug("Retrieved RSI", 
                        symbol=symbol, 
                        period=time_period,
                        records=len(data))
            return data
            
        except Exception as e:
            logger.error("Failed to get RSI", symbol=symbol, error=str(e))
            return pd.DataFrame()
    
    def get_macd(
        self,
        symbol: str,
        interval: str = 'daily',
        series_type: str = 'close'
    ) -> pd.DataFrame:
        """Get MACD indicator"""
        try:
            self._rate_limit()
            data, meta_data = self.indicators.get_macd(
                symbol=symbol,
                interval=interval,
                series_type=series_type
            )
            
            logger.debug("Retrieved MACD", symbol=symbol, records=len(data))
            return data
            
        except Exception as e:
            logger.error("Failed to get MACD", symbol=symbol, error=str(e))
            return pd.DataFrame()
    
    # === SECTOR PERFORMANCE ===
    
    def get_sector_performance(self) -> Dict[str, pd.DataFrame]:
        """Get sector performance data"""
        try:
            self._rate_limit()
            if self.sectors:
                data, meta_data = self.sectors.get_sector()
            else:
                logger.warning("Sector performance data not available")
                return {}
            
            # Parse different time periods
            results = {}
            for key, df in data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    results[key] = df
            
            logger.debug("Retrieved sector performance", periods=len(results))
            return results
            
        except Exception as e:
            logger.error("Failed to get sector performance", error=str(e))
            return {}
    
    # === ECONOMIC INDICATORS ===
    
    def get_economic_indicator(
        self,
        function: str,
        interval: str = 'monthly'
    ) -> pd.DataFrame:
        """
        Get economic indicators
        
        Available functions:
        - REAL_GDP, REAL_GDP_PER_CAPITA
        - TREASURY_YIELD, FEDERAL_FUNDS_RATE
        - CPI, INFLATION, RETAIL_SALES
        - DURABLES, UNEMPLOYMENT, NONFARM_PAYROLL
        """
        try:
            self._rate_limit()
            url = "https://www.alphavantage.co/query"
            params = {
                'function': function,
                'interval': interval,
                'apikey': self.api_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            # Convert to DataFrame
            if 'data' in data:
                df = pd.DataFrame(data['data'])
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    df = df.sort_index()
                    
                    # Convert value column to numeric
                    if 'value' in df.columns:
                        df['value'] = pd.to_numeric(df['value'], errors='coerce')
                
                logger.debug("Retrieved economic indicator", 
                           function=function, records=len(df))
                return df
            else:
                logger.warning("No data in economic indicator response", 
                             function=function, response=data)
                return pd.DataFrame()
                
        except Exception as e:
            logger.error("Failed to get economic indicator", 
                        function=function, error=str(e))
            return pd.DataFrame()
    
    # === NEWS AND SENTIMENT ===
    
    def get_news_sentiment(
        self,
        tickers: Optional[str] = None,
        topics: Optional[str] = None,
        time_from: Optional[str] = None,
        time_to: Optional[str] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """Get news sentiment data"""
        try:
            self._rate_limit()
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'apikey': self.api_key,
                'limit': limit
            }
            
            if tickers:
                params['tickers'] = tickers
            if topics:
                params['topics'] = topics
            if time_from:
                params['time_from'] = time_from
            if time_to:
                params['time_to'] = time_to
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'feed' in data:
                # Process news articles
                articles = []
                for article in data['feed']:
                    articles.append({
                        'title': article.get('title', ''),
                        'summary': article.get('summary', ''),
                        'url': article.get('url', ''),
                        'time_published': article.get('time_published', ''),
                        'authors': article.get('authors', []),
                        'source': article.get('source', ''),
                        'overall_sentiment_score': float(article.get('overall_sentiment_score', 0)),
                        'overall_sentiment_label': article.get('overall_sentiment_label', ''),
                        'ticker_sentiment': article.get('ticker_sentiment', [])
                    })
                
                # Calculate aggregate sentiment
                sentiment_scores = [a['overall_sentiment_score'] for a in articles]
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
                
                logger.debug("Retrieved news sentiment", 
                           articles=len(articles),
                           avg_sentiment=avg_sentiment)
                
                return {
                    'articles': articles,
                    'sentiment_summary': {
                        'avg_sentiment_score': avg_sentiment,
                        'article_count': len(articles),
                        'positive_articles': len([s for s in sentiment_scores if s > 0.1]),
                        'negative_articles': len([s for s in sentiment_scores if s < -0.1]),
                        'neutral_articles': len([s for s in sentiment_scores if -0.1 <= s <= 0.1])
                    }
                }
            else:
                logger.warning("No news data returned", response=data)
                return {'articles': [], 'sentiment_summary': {}}
                
        except Exception as e:
            logger.error("Failed to get news sentiment", error=str(e))
            return {'articles': [], 'sentiment_summary': {}}
    
    # === CRYPTO DATA ===
    
    def get_crypto_daily(
        self,
        symbol: str,
        market: str = 'USD'
    ) -> pd.DataFrame:
        """Get daily crypto data"""
        try:
            self._rate_limit()  
            data, meta_data = self.crypto.get_digital_currency_daily(
                symbol=symbol,
                market=market
            )
            
            # Clean column names
            data.columns = [col.split('(')[0].strip().lower() for col in data.columns]
            data.index.name = 'date'
            data = data.sort_index()
            
            logger.debug("Retrieved crypto daily data", 
                        symbol=symbol, records=len(data))
            return data
            
        except Exception as e:
            logger.error("Failed to get crypto daily data", 
                        symbol=symbol, error=str(e))
            return pd.DataFrame()
    
    # === UTILITY METHODS ===
    
    def search_endpoint(self, keywords: str) -> List[Dict[str, Any]]:
        """Search for symbols matching keywords"""
        try:
            self._rate_limit()
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'SYMBOL_SEARCH',
                'keywords': keywords,
                'apikey': self.api_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'bestMatches' in data:
                matches = []
                for match in data['bestMatches']:
                    matches.append({
                        'symbol': match.get('1. symbol', ''),
                        'name': match.get('2. name', ''),
                        'type': match.get('3. type', ''),
                        'region': match.get('4. region', ''),
                        'currency': match.get('8. currency', ''),
                        'match_score': float(match.get('9. matchScore', 0))
                    })
                
                logger.debug("Symbol search results", 
                           keywords=keywords, matches=len(matches))
                return matches
            
            return []
            
        except Exception as e:
            logger.error("Failed to search symbols", keywords=keywords, error=str(e))
            return []
    
    def get_api_status(self) -> Dict[str, Any]:
        """Check API status and remaining calls"""
        try:
            # Make a lightweight call to test API
            response = self.get_quote('AAPL')
            
            if response:
                return {
                    'status': 'active',
                    'last_successful_call': datetime.now().isoformat(),
                    'rate_limit_per_minute': 5  # Free tier limit
                }
            else:
                return {
                    'status': 'error',
                    'message': 'API call failed'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def __str__(self) -> str:
        return "AlphaVantageClient(rate_limit=5/min)"
    
    def __repr__(self) -> str:
        return self.__str__()