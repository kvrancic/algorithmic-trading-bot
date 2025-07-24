"""
Cryptocurrency Data Client for QuantumSentiment Trading Bot

Handles interactions with various crypto APIs for:
- Real-time crypto prices
- Historical OHLCV data
- Market data and metrics  
- DeFi and on-chain analytics
"""

import os
import time
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import structlog
import requests
import aiohttp

logger = structlog.get_logger(__name__)


class CryptoClient:
    """Multi-source cryptocurrency data client"""
    
    def __init__(self):
        """Initialize crypto client with multiple data sources"""
        
        # API endpoints
        self.coinpaprika_base = "https://api.coinpaprika.com/v1"
        self.coingecko_base = "https://api.coingecko.com/api/v3"
        self.binance_base = "https://api.binance.com/api/v3"
        
        # Rate limiting
        self.last_call_times = {
            'coinpaprika': 0,
            'coingecko': 0, 
            'binance': 0
        }
        
        # Rate limits (requests per second)
        self.rate_limits = {
            'coinpaprika': 0.1,  # 10 requests per second
            'coingecko': 0.05,   # ~20 requests per second  
            'binance': 0.1       # 10 requests per second
        }
        
        # Symbol mappings
        self.symbol_map = {
            'BTC': {'coinpaprika': 'btc-bitcoin', 'coingecko': 'bitcoin', 'binance': 'BTCUSDT'},
            'ETH': {'coinpaprika': 'eth-ethereum', 'coingecko': 'ethereum', 'binance': 'ETHUSDT'},
            'BNB': {'coinpaprika': 'bnb-binance-coin', 'coingecko': 'binancecoin', 'binance': 'BNBUSDT'},
            'ADA': {'coinpaprika': 'ada-cardano', 'coingecko': 'cardano', 'binance': 'ADAUSDT'},
            'SOL': {'coinpaprika': 'sol-solana', 'coingecko': 'solana', 'binance': 'SOLUSDT'},
            'MATIC': {'coinpaprika': 'matic-polygon', 'coingecko': 'matic-network', 'binance': 'MATICUSDT'},
            'DOT': {'coinpaprika': 'dot-polkadot', 'coingecko': 'polkadot', 'binance': 'DOTUSDT'},
            'AVAX': {'coinpaprika': 'avax-avalanche', 'coingecko': 'avalanche-2', 'binance': 'AVAXUSDT'},
            'LINK': {'coinpaprika': 'link-chainlink', 'coingecko': 'chainlink', 'binance': 'LINKUSDT'},
            'DOGE': {'coinpaprika': 'doge-dogecoin', 'coingecko': 'dogecoin', 'binance': 'DOGEUSDT'}
        }
        
        logger.info("Crypto client initialized with multiple sources")
    
    def _rate_limit(self, source: str):
        """Enforce rate limiting for specific source"""
        elapsed = time.time() - self.last_call_times[source]
        min_interval = self.rate_limits[source]
        
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            time.sleep(sleep_time)
        
        self.last_call_times[source] = time.time()
    
    # === COINPAPRIKA API ===
    
    def _get_coinpaprika_data(self, endpoint: str) -> Dict[str, Any]:
        """Make request to CoinPaprika API"""
        try:
            self._rate_limit('coinpaprika')
            
            url = f"{self.coinpaprika_base}/{endpoint}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error("CoinPaprika API error", endpoint=endpoint, error=str(e))
            return {}
    
    def get_crypto_quote_coinpaprika(self, symbol: str) -> Dict[str, Any]:
        """Get crypto quote from CoinPaprika"""
        if symbol not in self.symbol_map:
            logger.warning("Symbol not supported", symbol=symbol)
            return {}
        
        coin_id = self.symbol_map[symbol]['coinpaprika']
        data = self._get_coinpaprika_data(f"tickers/{coin_id}")
        
        if data:
            quotes = data.get('quotes', {}).get('USD', {})
            return {
                'symbol': symbol,
                'price': quotes.get('price', 0),
                'volume_24h': quotes.get('volume_24h', 0),
                'percent_change_1h': quotes.get('percent_change_1h', 0),
                'percent_change_24h': quotes.get('percent_change_24h', 0),
                'percent_change_7d': quotes.get('percent_change_7d', 0),
                'market_cap': quotes.get('market_cap', 0),
                'last_updated': data.get('last_updated'),
                'source': 'coinpaprika'
            }
        
        return {}
    
    def get_crypto_historical_coinpaprika(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """Get historical data from CoinPaprika"""
        if symbol not in self.symbol_map:
            return pd.DataFrame()
        
        coin_id = self.symbol_map[symbol]['coinpaprika']
        
        # Format dates
        start_str = start.strftime('%Y-%m-%d')
        end_str = (end or datetime.now()).strftime('%Y-%m-%d')
        
        endpoint = f"tickers/{coin_id}/historical?start={start_str}&end={end_str}&interval={interval}"
        data = self._get_coinpaprika_data(endpoint)
        
        if data:
            df = pd.DataFrame(data)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # Extract price data
                df['price'] = df['price'].astype(float)
                df['volume_24h'] = df['volume_24h'].astype(float)
                df['market_cap'] = df['market_cap'].astype(float)
                
                logger.debug("Retrieved CoinPaprika historical data", 
                           symbol=symbol, records=len(df))
                return df
        
        return pd.DataFrame()
    
    # === COINGECKO API ===
    
    def _get_coingecko_data(self, endpoint: str, params: Dict = None) -> Dict[str, Any]:
        """Make request to CoinGecko API"""
        try:
            self._rate_limit('coingecko')
            
            url = f"{self.coingecko_base}/{endpoint}"
            response = requests.get(url, params=params or {}, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error("CoinGecko API error", endpoint=endpoint, error=str(e))
            return {}
    
    def get_crypto_quote_coingecko(self, symbol: str) -> Dict[str, Any]:
        """Get crypto quote from CoinGecko"""
        if symbol not in self.symbol_map:
            return {}
        
        coin_id = self.symbol_map[symbol]['coingecko']
        params = {
            'ids': coin_id,
            'vs_currencies': 'usd',
            'include_market_cap': 'true',
            'include_24hr_vol': 'true',
            'include_24hr_change': 'true'
        }
        
        data = self._get_coingecko_data("simple/price", params)
        
        if coin_id in data:
            coin_data = data[coin_id]
            return {
                'symbol': symbol,
                'price': coin_data.get('usd', 0),
                'market_cap': coin_data.get('usd_market_cap', 0),
                'volume_24h': coin_data.get('usd_24h_vol', 0),
                'percent_change_24h': coin_data.get('usd_24h_change', 0),
                'last_updated': datetime.now().isoformat(),
                'source': 'coingecko'
            }
        
        return {}
    
    def get_crypto_historical_coingecko(
        self,
        symbol: str,
        days: int = 30,
        interval: str = 'daily'
    ) -> pd.DataFrame:
        """Get historical data from CoinGecko"""
        if symbol not in self.symbol_map:
            return pd.DataFrame()
        
        coin_id = self.symbol_map[symbol]['coingecko']
        
        # Map intervals
        interval_map = {
            'daily': 'daily',
            'hourly': 'hourly' if days <= 90 else 'daily'
        }
        
        endpoint = f"coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': interval_map.get(interval, 'daily')
        }
        
        data = self._get_coingecko_data(endpoint, params)
        
        if data and 'prices' in data:
            # Convert to DataFrame
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            market_caps = data.get('market_caps', [])
            
            df_data = []
            for i, (timestamp, price) in enumerate(prices):
                row = {
                    'timestamp': pd.to_datetime(timestamp, unit='ms'),
                    'price': price,
                    'volume': volumes[i][1] if i < len(volumes) else 0,
                    'market_cap': market_caps[i][1] if i < len(market_caps) else 0
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                df = df.sort_index()
                
                logger.debug("Retrieved CoinGecko historical data", 
                           symbol=symbol, records=len(df))
                return df
        
        return pd.DataFrame()
    
    # === BINANCE API ===
    
    def _get_binance_data(self, endpoint: str, params: Dict = None) -> Dict[str, Any]:
        """Make request to Binance API"""
        try:
            self._rate_limit('binance')
            
            url = f"{self.binance_base}/{endpoint}"
            response = requests.get(url, params=params or {}, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error("Binance API error", endpoint=endpoint, error=str(e))
            return {}
    
    def get_crypto_quote_binance(self, symbol: str) -> Dict[str, Any]:
        """Get crypto quote from Binance"""
        if symbol not in self.symbol_map:
            return {}
        
        binance_symbol = self.symbol_map[symbol]['binance']
        
        # Get 24hr ticker statistics
        data = self._get_binance_data(f"ticker/24hr", {'symbol': binance_symbol})
        
        if data and 'symbol' in data:
            return {
                'symbol': symbol,
                'price': float(data.get('lastPrice', 0)),
                'volume_24h': float(data.get('volume', 0)),
                'percent_change_24h': float(data.get('priceChangePercent', 0)),
                'high_24h': float(data.get('highPrice', 0)),
                'low_24h': float(data.get('lowPrice', 0)),
                'open_24h': float(data.get('openPrice', 0)),
                'last_updated': datetime.now().isoformat(),
                'source': 'binance'
            }
        
        return {}
    
    def get_crypto_klines_binance(
        self,
        symbol: str,
        interval: str = '1d',
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """Get kline/candlestick data from Binance"""
        if symbol not in self.symbol_map:
            return pd.DataFrame()
        
        binance_symbol = self.symbol_map[symbol]['binance']
        
        params = {
            'symbol': binance_symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        data = self._get_binance_data("klines", params)
        
        if data and isinstance(data, list):
            # Convert to DataFrame
            columns = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]
            
            df = pd.DataFrame(data, columns=columns)
            
            if not df.empty:
                # Convert timestamp and set as index
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Convert numeric columns
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col])
                
                # Keep only OHLCV data
                df = df[numeric_cols]
                df = df.sort_index()
                
                logger.debug("Retrieved Binance klines", 
                           symbol=symbol, records=len(df))
                return df
        
        return pd.DataFrame()
    
    # === UNIFIED INTERFACE ===
    
    def get_crypto_quote(
        self,
        symbol: str,
        preferred_source: str = 'coingecko'
    ) -> Dict[str, Any]:
        """
        Get crypto quote with fallback sources
        
        Args:
            symbol: Crypto symbol (BTC, ETH, etc.)
            preferred_source: Preferred data source
            
        Returns:
            Quote data with unified format
        """
        sources = [preferred_source, 'coingecko', 'coinpaprika', 'binance']
        sources = list(dict.fromkeys(sources))  # Remove duplicates
        
        for source in sources:
            try:
                if source == 'coingecko':
                    quote = self.get_crypto_quote_coingecko(symbol)
                elif source == 'coinpaprika':
                    quote = self.get_crypto_quote_coinpaprika(symbol)
                elif source == 'binance':
                    quote = self.get_crypto_quote_binance(symbol)
                else:
                    continue
                
                if quote and quote.get('price', 0) > 0:
                    logger.debug("Retrieved crypto quote", 
                               symbol=symbol, 
                               source=source,
                               price=quote['price'])
                    return quote
                    
            except Exception as e:
                logger.warning("Failed to get quote from source", 
                             symbol=symbol, source=source, error=str(e))
                continue
        
        logger.error("Failed to get quote from all sources", symbol=symbol)
        return {}
    
    def get_crypto_historical(
        self,
        symbol: str,
        days: int = 30,
        interval: str = 'daily',
        preferred_source: str = 'coingecko'
    ) -> pd.DataFrame:
        """Get historical data with fallback sources"""
        
        sources = [preferred_source, 'coingecko', 'binance', 'coinpaprika']
        sources = list(dict.fromkeys(sources))
        
        for source in sources:
            try:
                if source == 'coingecko':
                    df = self.get_crypto_historical_coingecko(symbol, days, interval)
                elif source == 'binance':
                    # Map interval for Binance
                    binance_interval = {'daily': '1d', 'hourly': '1h'}.get(interval, '1d')
                    df = self.get_crypto_klines_binance(symbol, binance_interval, days*24 if interval == 'hourly' else days)
                elif source == 'coinpaprika':
                    start_date = datetime.now() - timedelta(days=days)
                    df = self.get_crypto_historical_coinpaprika(symbol, start_date)
                else:
                    continue
                
                if not df.empty:
                    logger.debug("Retrieved crypto historical data", 
                               symbol=symbol, 
                               source=source,
                               records=len(df))
                    return df
                    
            except Exception as e:
                logger.warning("Failed to get historical data from source", 
                             symbol=symbol, source=source, error=str(e))
                continue
        
        logger.error("Failed to get historical data from all sources", symbol=symbol)
        return pd.DataFrame()
    
    def get_multiple_quotes(
        self,
        symbols: List[str],
        preferred_source: str = 'coingecko'
    ) -> Dict[str, Dict[str, Any]]:
        """Get quotes for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            try:
                quote = self.get_crypto_quote(symbol, preferred_source)
                if quote:
                    results[symbol] = quote
                else:
                    results[symbol] = {}
                    
                # Small delay to respect rate limits
                time.sleep(0.1)
                
            except Exception as e:
                logger.error("Failed to get quote", symbol=symbol, error=str(e))
                results[symbol] = {}
        
        logger.info("Retrieved multiple crypto quotes", 
                   requested=len(symbols), 
                   successful=len([r for r in results.values() if r]))
        
        return results
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get overall crypto market overview"""
        try:
            # Get global market data from CoinGecko
            data = self._get_coingecko_data("global")
            
            if data and 'data' in data:
                global_data = data['data']
                return {
                    'total_market_cap_usd': global_data.get('total_market_cap', {}).get('usd', 0),
                    'total_volume_24h_usd': global_data.get('total_volume', {}).get('usd', 0),
                    'bitcoin_dominance': global_data.get('market_cap_percentage', {}).get('btc', 0),
                    'ethereum_dominance': global_data.get('market_cap_percentage', {}).get('eth', 0),
                    'active_cryptocurrencies': global_data.get('active_cryptocurrencies', 0),
                    'markets': global_data.get('markets', 0),
                    'market_cap_change_24h': global_data.get('market_cap_change_percentage_24h_usd', 0),
                    'updated_at': global_data.get('updated_at')
                }
            
        except Exception as e:
            logger.error("Failed to get market overview", error=str(e))
        
        return {}
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported cryptocurrency symbols"""
        return list(self.symbol_map.keys())
    
    def is_symbol_supported(self, symbol: str) -> bool:
        """Check if symbol is supported"""
        return symbol.upper() in self.symbol_map
    
    def __str__(self) -> str:
        return f"CryptoClient(sources=3, symbols={len(self.symbol_map)})"
    
    def __repr__(self) -> str:
        return self.__str__()