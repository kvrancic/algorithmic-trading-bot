"""
Alpaca API Client for QuantumSentiment Trading Bot

Handles all interactions with Alpaca Markets API for:
- Market data (stocks, crypto)
- Account information
- Order management
- Portfolio tracking
"""

import os
import asyncio
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import pandas as pd
import structlog
from alpaca_trade_api import REST
from alpaca_trade_api.entity import Account, Position, Order, Asset
from alpaca_trade_api.rest import TimeFrame

logger = structlog.get_logger(__name__)


class AlpacaClient:
    """Alpaca API client with enhanced functionality for trading bot"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        paper: bool = True
    ):
        """
        Initialize Alpaca client
        
        Args:
            api_key: Alpaca API key (defaults to env var)
            api_secret: Alpaca API secret (defaults to env var)
            base_url: Base URL (defaults to paper trading)
            paper: Whether to use paper trading
        """
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.api_secret = api_secret or os.getenv('ALPACA_API_SECRET')
        
        if paper:
            self.base_url = 'https://paper-api.alpaca.markets'
        else:
            self.base_url = base_url or os.getenv('ALPACA_BASE_URL', 'https://api.alpaca.markets')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials not provided")
        
        # Initialize REST API
        self.api = REST(
            self.api_key,
            self.api_secret, 
            self.base_url
        )
        
        self.paper_trading = paper
        logger.info("Alpaca client initialized", 
                   paper_trading=self.paper_trading,
                   base_url=self.base_url)
    
    # === ACCOUNT INFORMATION ===
    
    def get_account(self) -> Account:
        """Get account information"""
        try:
            account = self.api.get_account()
            logger.debug("Retrieved account info",
                        buying_power=float(account.buying_power),
                        equity=float(account.equity),
                        status=account.status)
            return account
        except Exception as e:
            logger.error("Failed to get account info", error=str(e))
            raise
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        account = self.get_account()
        return float(account.portfolio_value)
    
    def get_buying_power(self) -> float:
        """Get available buying power"""
        account = self.get_account()
        return float(account.buying_power)
    
    def get_positions(self) -> List[Position]:
        """Get all current positions"""
        try:
            positions = self.api.list_positions()
            logger.debug("Retrieved positions", count=len(positions))
            return positions
        except Exception as e:
            logger.error("Failed to get positions", error=str(e))
            raise
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol"""
        try:
            position = self.api.get_position(symbol)
            return position
        except Exception as e:
            if "position does not exist" in str(e).lower():
                return None
            logger.error("Failed to get position", symbol=symbol, error=str(e))
            raise
    
    # === MARKET DATA ===
    
    def get_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """Get latest quote for symbol"""
        try:
            quote = self.api.get_latest_quote(symbol)
            return {
                'symbol': symbol,
                'bid': float(quote.bid_price),
                'ask': float(quote.ask_price),
                'bid_size': quote.bid_size,
                'ask_size': quote.ask_size,
                'timestamp': quote.timestamp
            }
        except Exception as e:
            logger.error("Failed to get quote", symbol=symbol, error=str(e))
            raise
    
    def get_latest_trade(self, symbol: str) -> Dict[str, Any]:
        """Get latest trade for symbol"""
        try:
            trade = self.api.get_latest_trade(symbol)
            return {
                'symbol': symbol,
                'price': float(trade.price),
                'size': trade.size,
                'timestamp': trade.timestamp
            }
        except Exception as e:
            logger.error("Failed to get latest trade", symbol=symbol, error=str(e))
            raise
    
    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol for Alpaca API
        
        For crypto symbols, Alpaca bars API uses format without separators (BTCUSD) not BTC-USD or BTC/USD
        """
        # Convert crypto symbols from BTC-USD format to BTCUSD format (remove dash for bars API)
        if '-' in symbol and any(crypto in symbol for crypto in ['BTC', 'ETH', 'LTC', 'BCH', 'DOGE', 'SHIB', 'AVAX', 'MATIC']):
            return symbol.replace('-', '')  # BTC-USD becomes BTCUSD
        return symbol

    def get_bars(
        self,
        symbol: str,
        timeframe: str = '1Hour',
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get historical bars data
        
        Args:
            symbol: Stock symbol
            timeframe: Bar timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)
            start: Start datetime
            end: End datetime
            limit: Max number of bars
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Normalize symbol for Alpaca API (convert BTC-USD to BTCUSD etc.)
            normalized_symbol = self._normalize_symbol(symbol)
            
            # No URL encoding needed since crypto symbols no longer use / separator
            api_symbol = normalized_symbol
                
            logger.debug("Symbol normalized", 
                        original=symbol, 
                        normalized=normalized_symbol)
            
            # Map timeframe string to Alpaca TimeFrame
            timeframe_map = {
                '1Min': TimeFrame.Minute,
                '5Min': TimeFrame(5, TimeFrame.Minute),
                '15Min': TimeFrame(15, TimeFrame.Minute),
                '1Hour': TimeFrame.Hour,
                '1Day': TimeFrame.Day
            }
            
            tf = timeframe_map.get(timeframe, TimeFrame.Hour)
            
            # Default to last 30 days if no start time
            if start is None:
                start = datetime.now() - timedelta(days=30)
            
            # For paper trading accounts, omit end parameter to automatically 
            # default to 15 minutes ago and avoid "subscription does not permit querying recent SIP data" error
            # Per Alpaca forum: leaving end=None defaults to exactly what we want for paper accounts
            if end is not None:
                fifteen_minutes_ago = datetime.now() - timedelta(minutes=15)
                if isinstance(end, datetime) and end > fifteen_minutes_ago:
                    logger.debug("Omitting end time to avoid recent data restriction", 
                               requested_end=end, note="Will default to 15 minutes ago")
                    end = None
            
            # Format datetime for Alpaca API - use RFC3339/ISO8601 format
            # For daily data, use date format; for intraday, use datetime format
            is_daily = tf == TimeFrame.Day
            
            if is_daily:
                # Daily timeframes use date format: YYYY-MM-DD
                start_str = start.strftime('%Y-%m-%d') if isinstance(start, datetime) else start
                end_str = end.strftime('%Y-%m-%d') if isinstance(end, datetime) and end else None
            else:
                # Intraday timeframes use RFC3339 format: YYYY-MM-DDTHH:MM:SSZ
                start_str = start.strftime('%Y-%m-%dT%H:%M:%SZ') if isinstance(start, datetime) else start
                end_str = end.strftime('%Y-%m-%dT%H:%M:%SZ') if isinstance(end, datetime) and end else None
            
            # Build API parameters - omit end if None to use Alpaca's default (15 minutes ago)
            api_params = {
                'start': start_str,
                'limit': limit,
                'adjustment': 'raw'
            }
            
            # Only include end parameter if we have a value
            if end_str is not None:
                api_params['end'] = end_str
                
            bars = self.api.get_bars(api_symbol, tf, **api_params)
            
            # Convert to DataFrame
            data = []
            for bar in bars:
                data.append({
                    'timestamp': bar.t,  # Alpaca uses 't' for timestamp
                    'open': float(bar.o),   # Alpaca uses 'o' for open
                    'high': float(bar.h),   # Alpaca uses 'h' for high
                    'low': float(bar.l),    # Alpaca uses 'l' for low
                    'close': float(bar.c),  # Alpaca uses 'c' for close
                    'volume': bar.v  # Alpaca uses 'v' for volume
                })
            
            df = pd.DataFrame(data)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
            
            logger.debug("Retrieved bars", 
                        symbol=symbol, 
                        timeframe=timeframe,
                        count=len(df))
            return df
            
        except Exception as e:
            logger.error("Failed to get bars", 
                        symbol=symbol, 
                        timeframe=timeframe,
                        error=str(e))
            raise
    
    def get_multiple_bars(
        self,
        symbols: List[str],
        timeframe: str = '1Hour',
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> Dict[str, pd.DataFrame]:
        """Get bars for multiple symbols"""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_bars(symbol, timeframe, start, end, limit)
            except Exception as e:
                logger.warning("Failed to get bars for symbol", 
                             symbol=symbol, error=str(e))
                results[symbol] = pd.DataFrame()
        return results
    
    # === ORDER MANAGEMENT ===
    
    def submit_order(
        self,
        symbol: str,
        qty: Union[int, float],
        side: str,
        type: str = 'market',
        time_in_force: str = 'day',
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None
    ) -> Order:
        """
        Submit an order
        
        Args:
            symbol: Stock symbol
            qty: Quantity to trade
            side: 'buy' or 'sell'
            type: Order type ('market', 'limit', 'stop', 'stop_limit')
            time_in_force: 'day', 'gtc', 'ioc', 'fok'
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            client_order_id: Custom order ID
            
        Returns:
            Order object
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                stop_price=stop_price,
                client_order_id=client_order_id
            )
            
            logger.info("Order submitted",
                       symbol=symbol,
                       side=side,
                       qty=qty,
                       type=type,
                       order_id=order.id)
            return order
            
        except Exception as e:
            logger.error("Failed to submit order",
                        symbol=symbol,
                        side=side,
                        qty=qty,
                        error=str(e))
            raise
    
    def get_orders(
        self,
        status: str = 'all',
        limit: int = 100,
        direction: str = 'desc'
    ) -> List[Order]:
        """Get orders with optional filtering"""
        try:
            orders = self.api.list_orders(
                status=status,
                limit=limit,
                direction=direction
            )
            logger.debug("Retrieved orders", count=len(orders), status=status)
            return orders
        except Exception as e:
            logger.error("Failed to get orders", error=str(e))
            raise
    
    def cancel_order(self, order_id: str) -> None:
        """Cancel an order"""
        try:
            self.api.cancel_order(order_id)
            logger.info("Order cancelled", order_id=order_id)
        except Exception as e:
            logger.error("Failed to cancel order", order_id=order_id, error=str(e))
            raise
    
    def cancel_all_orders(self) -> None:
        """Cancel all open orders"""
        try:
            self.api.cancel_all_orders()
            logger.info("All orders cancelled")
        except Exception as e:
            logger.error("Failed to cancel all orders", error=str(e))
            raise
    
    # === UTILITY METHODS ===
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error("Failed to check market status", error=str(e))
            return False
    
    def get_market_calendar(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[Dict]:
        """Get market calendar"""
        try:
            if start is None:
                start = datetime.now()
            if end is None:
                end = start + timedelta(days=30)
                
            calendar = self.api.get_calendar(start=start, end=end)
            return [
                {
                    'date': day.date,
                    'open': day.open,
                    'close': day.close
                }
                for day in calendar
            ]
        except Exception as e:
            logger.error("Failed to get market calendar", error=str(e))
            raise
    
    def get_asset_info(self, symbol: str) -> Optional[Asset]:
        """Get asset information"""
        try:
            asset = self.api.get_asset(symbol)
            return asset
        except Exception as e:
            logger.warning("Failed to get asset info", symbol=symbol, error=str(e))
            return None
    
    def is_tradable(self, symbol: str) -> bool:
        """Check if asset is tradable"""
        asset = self.get_asset_info(symbol)
        if asset:
            return asset.tradable and asset.status == 'active'
        return False
    
    def get_portfolio_history(
        self,
        period: str = '1D',
        timeframe: str = '1Min'
    ) -> Dict[str, Any]:
        """Get portfolio performance history"""
        try:
            history = self.api.get_portfolio_history(
                period=period,
                timeframe=timeframe
            )
            return {
                'timestamp': history.timestamp,
                'equity': history.equity,
                'profit_loss': history.profit_loss,
                'profit_loss_pct': history.profit_loss_pct
            }
        except Exception as e:
            logger.error("Failed to get portfolio history", error=str(e))
            raise
    
    # === ASYNC METHODS ===
    
    async def stream_quotes(self, symbols: List[str], callback):
        """Stream real-time quotes (placeholder for future implementation)"""
        logger.info("Quote streaming not implemented yet", symbols=symbols)
        # TODO: Implement real-time streaming using Alpaca's WebSocket API
        pass
    
    def __str__(self) -> str:
        return f"AlpacaClient(paper={self.paper_trading})"
    
    def __repr__(self) -> str:
        return self.__str__()