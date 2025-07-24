"""
Alpaca Broker Integration

Enhanced Alpaca broker integration with:
- Unified broker interface
- Paper trading support
- Advanced order management
- Real-time data streaming
- Position and account synchronization
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import asyncio
import pandas as pd
import structlog

from ..data.alpaca_client import AlpacaClient
from .order_manager import OrderManager, Order, OrderStatus, OrderType, OrderSide, TimeInForce
from .position_tracker import PositionTracker
from .account_monitor import AccountMonitor

logger = structlog.get_logger(__name__)


class AlpacaBroker:
    """Enhanced Alpaca broker with integrated order management and monitoring"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper_trading: bool = True,
        order_manager: Optional[OrderManager] = None,
        position_tracker: Optional[PositionTracker] = None,
        account_monitor: Optional[AccountMonitor] = None
    ):
        """
        Initialize Alpaca broker
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            paper_trading: Use paper trading mode
            order_manager: Order management system
            position_tracker: Position tracking system
            account_monitor: Account monitoring system
        """
        
        # Initialize Alpaca client
        self.client = AlpacaClient(
            api_key=api_key,
            api_secret=api_secret,
            paper=paper_trading
        )
        
        # Integration components
        self.order_manager = order_manager
        self.position_tracker = position_tracker
        self.account_monitor = account_monitor
        
        # Broker state
        self.paper_trading = paper_trading
        self.is_connected = False
        
        # Set up integrations
        self._setup_integrations()
        
        logger.info("Alpaca broker initialized",
                   paper_trading=paper_trading,
                   has_order_manager=order_manager is not None,
                   has_position_tracker=position_tracker is not None,
                   has_account_monitor=account_monitor is not None)
    
    def _setup_integrations(self) -> None:
        """Set up component integrations"""
        if self.order_manager:
            self.order_manager.set_broker(self)
        
        if self.position_tracker:
            self.position_tracker.set_broker(self)
        
        if self.account_monitor:
            self.account_monitor.set_broker(self)
            if self.position_tracker:
                self.account_monitor.set_position_tracker(self.position_tracker)
    
    async def connect(self) -> bool:
        """
        Connect to Alpaca and verify credentials
        
        Returns:
            True if connection successful
        """
        try:
            # Test connection by getting account info
            account = await self.get_account()
            if account:
                self.is_connected = True
                logger.info("Connected to Alpaca successfully",
                           account_status=account.status,
                           equity=float(account.equity),
                           buying_power=float(account.buying_power))
                return True
            else:
                self.is_connected = False
                return False
                
        except Exception as e:
            self.is_connected = False
            logger.error("Failed to connect to Alpaca", error=str(e))
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from broker"""
        self.is_connected = False
        logger.info("Disconnected from Alpaca")
    
    # === ACCOUNT METHODS ===
    
    async def get_account(self):
        """Get account information"""
        try:
            return self.client.get_account()
        except Exception as e:
            logger.error("Failed to get account", error=str(e))
            return None
    
    async def get_buying_power(self) -> float:
        """Get available buying power"""
        try:
            return self.client.get_buying_power()
        except Exception as e:
            logger.error("Failed to get buying power", error=str(e))
            return 0.0
    
    async def get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        try:
            return self.client.get_portfolio_value()
        except Exception as e:
            logger.error("Failed to get portfolio value", error=str(e))
            return 0.0
    
    # === POSITION METHODS ===
    
    async def get_positions(self):
        """Get all current positions"""
        try:
            return self.client.get_positions()
        except Exception as e:
            logger.error("Failed to get positions", error=str(e))
            return []
    
    async def get_position(self, symbol: str):
        """Get position for specific symbol"""
        try:
            return self.client.get_position(symbol)
        except Exception as e:
            logger.error("Failed to get position", symbol=symbol, error=str(e))
            return None
    
    # === ORDER METHODS ===
    
    async def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: Union[int, float],
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
        client_order_id: Optional[str] = None
    ):
        """
        Submit order to Alpaca
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            order_type: Order type
            limit_price: Limit price
            stop_price: Stop price
            time_in_force: Time in force
            client_order_id: Custom order ID
            
        Returns:
            Alpaca order object
        """
        try:
            # Submit to Alpaca
            alpaca_order = self.client.submit_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                type=order_type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                stop_price=stop_price,
                client_order_id=client_order_id
            )
            
            # If we have an order manager, track the order
            if self.order_manager and alpaca_order:
                await self._sync_order_with_manager(alpaca_order)
            
            logger.info("Order submitted to Alpaca",
                       symbol=symbol,
                       side=side,
                       quantity=quantity,
                       order_type=order_type,
                       order_id=alpaca_order.id)
            
            return alpaca_order
            
        except Exception as e:
            logger.error("Failed to submit order",
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        error=str(e))
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation successful
        """
        try:
            self.client.cancel_order(order_id)
            
            # Update order manager if available
            if self.order_manager:
                await self.order_manager.cancel_order(order_id, "broker_cancelled")
            
            logger.info("Order cancelled", order_id=order_id)
            return True
            
        except Exception as e:
            logger.error("Failed to cancel order", order_id=order_id, error=str(e))
            return False
    
    async def get_orders(self, status: str = "all", limit: int = 100):
        """Get orders from Alpaca"""
        try:
            return self.client.get_orders(status=status, limit=limit)
        except Exception as e:
            logger.error("Failed to get orders", error=str(e))
            return []
    
    async def modify_order(
        self,
        order_id: str,
        quantity: Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> bool:
        """
        Modify an existing order
        
        Args:
            order_id: Order to modify
            quantity: New quantity
            limit_price: New limit price
            stop_price: New stop price
            
        Returns:
            True if modification successful
        """
        try:
            # Alpaca doesn't support direct order modification
            # We need to cancel and resubmit
            
            # Get current order
            current_orders = await self.get_orders(status="open")
            target_order = None
            
            for order in current_orders:
                if order.id == order_id:
                    target_order = order
                    break
            
            if not target_order:
                logger.error("Order not found for modification", order_id=order_id)
                return False
            
            # Cancel current order
            cancel_success = await self.cancel_order(order_id)
            if not cancel_success:
                return False
            
            # Submit new order with modified parameters
            new_order = await self.submit_order(
                symbol=target_order.symbol,
                side=target_order.side,
                quantity=quantity or float(target_order.qty),
                order_type=target_order.order_type,
                limit_price=limit_price or (float(target_order.limit_price) if target_order.limit_price else None),
                stop_price=stop_price or (float(target_order.stop_price) if target_order.stop_price else None),
                time_in_force=target_order.time_in_force,
                client_order_id=f"modified_{target_order.client_order_id or target_order.id}"
            )
            
            logger.info("Order modified via cancel/resubmit",
                       original_order_id=order_id,
                       new_order_id=new_order.id if new_order else None)
            
            return new_order is not None
            
        except Exception as e:
            logger.error("Failed to modify order", order_id=order_id, error=str(e))
            return False
    
    # === MARKET DATA METHODS ===
    
    async def get_latest_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest quote for symbol"""
        try:
            return self.client.get_latest_quote(symbol)
        except Exception as e:
            logger.error("Failed to get latest quote", symbol=symbol, error=str(e))
            return None
    
    async def get_latest_trade(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest trade for symbol"""
        try:
            return self.client.get_latest_trade(symbol)
        except Exception as e:
            logger.error("Failed to get latest trade", symbol=symbol, error=str(e))
            return None
    
    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Hour",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Get historical bars"""
        try:
            return self.client.get_bars(symbol, timeframe, start, end, limit)
        except Exception as e:
            logger.error("Failed to get bars", symbol=symbol, error=str(e))
            return pd.DataFrame()
    
    async def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get latest prices for multiple symbols
        
        Args:
            symbols: List of symbols
            
        Returns:
            Dictionary of symbol -> price
        """
        prices = {}
        
        for symbol in symbols:
            try:
                trade_data = await self.get_latest_trade(symbol)
                if trade_data:
                    prices[symbol] = trade_data['price']
                else:
                    # Fallback to quote data
                    quote_data = await self.get_latest_quote(symbol)
                    if quote_data:
                        prices[symbol] = (quote_data['bid'] + quote_data['ask']) / 2
            except Exception as e:
                logger.warning("Failed to get price", symbol=symbol, error=str(e))
        
        return prices
    
    # === UTILITY METHODS ===
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            return self.client.is_market_open()
        except Exception as e:
            logger.error("Failed to check market status", error=str(e))
            return False
    
    def is_tradable(self, symbol: str) -> bool:
        """Check if symbol is tradable"""
        try:
            return self.client.is_tradable(symbol)
        except Exception as e:
            logger.error("Failed to check if symbol is tradable", symbol=symbol, error=str(e))
            return False
    
    # === SYNCHRONIZATION METHODS ===
    
    async def sync_orders(self) -> None:
        """Synchronize orders with order manager"""
        if not self.order_manager:
            return
        
        try:
            # Get all orders from Alpaca
            alpaca_orders = await self.get_orders(status="all", limit=500)
            
            for alpaca_order in alpaca_orders:
                await self._sync_order_with_manager(alpaca_order)
                
            logger.info("Orders synchronized", count=len(alpaca_orders))
            
        except Exception as e:
            logger.error("Failed to sync orders", error=str(e))
    
    async def sync_positions(self) -> None:
        """Synchronize positions with position tracker"""
        if not self.position_tracker:
            return
        
        try:
            await self.position_tracker.sync_positions_with_broker()
            logger.info("Positions synchronized")
            
        except Exception as e:
            logger.error("Failed to sync positions", error=str(e))
    
    async def sync_account(self) -> None:
        """Synchronize account data with account monitor"""
        if not self.account_monitor:
            return
        
        try:
            await self.account_monitor.take_snapshot()
            logger.info("Account data synchronized")
            
        except Exception as e:
            logger.error("Failed to sync account", error=str(e))
    
    async def full_sync(self) -> None:
        """Perform full synchronization of all data"""
        logger.info("Starting full broker synchronization")
        
        await asyncio.gather(
            self.sync_orders(),
            self.sync_positions(),
            self.sync_account(),
            return_exceptions=True
        )
        
        logger.info("Full broker synchronization completed")
    
    # === PRIVATE METHODS ===
    
    async def _sync_order_with_manager(self, alpaca_order) -> None:
        """Sync single Alpaca order with order manager"""
        if not self.order_manager:
            return
        
        try:
            # Check if order already exists in manager
            existing_order = self.order_manager.get_order(alpaca_order.id)
            
            if not existing_order:
                # Create new order in manager
                order_side = OrderSide.BUY if alpaca_order.side == "buy" else OrderSide.SELL
                order_type = self._map_alpaca_order_type(alpaca_order.order_type)
                time_in_force = self._map_alpaca_time_in_force(alpaca_order.time_in_force)
                
                managed_order = await self.order_manager.create_order(
                    symbol=alpaca_order.symbol,
                    side=order_side,
                    quantity=float(alpaca_order.qty),
                    order_type=order_type,
                    limit_price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
                    stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
                    time_in_force=time_in_force,
                    client_order_id=alpaca_order.client_order_id
                )
                
                # Store Alpaca order ID reference
                managed_order.tags['alpaca_order_id'] = alpaca_order.id
            else:
                managed_order = existing_order
            
            # Update order status based on Alpaca status
            alpaca_status = self._map_alpaca_order_status(alpaca_order.status)
            if managed_order.status != alpaca_status:
                await self.order_manager._update_order_status(managed_order, alpaca_status)
            
            # Process fills if any
            if hasattr(alpaca_order, 'filled_qty') and float(alpaca_order.filled_qty) > 0:
                filled_qty = float(alpaca_order.filled_qty)
                avg_fill_price = float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else 0
                
                # Check if we need to process new fills
                if filled_qty > managed_order.filled_quantity:
                    new_fill_qty = filled_qty - managed_order.filled_quantity
                    
                    await self.order_manager.process_fill(
                        order_id=managed_order.order_id,
                        fill_quantity=new_fill_qty,
                        fill_price=avg_fill_price,
                        venue="ALPACA"
                    )
                    
                    # Also update position tracker if available
                    if self.position_tracker:
                        quantity_change = new_fill_qty if managed_order.side == OrderSide.BUY else -new_fill_qty
                        
                        await self.position_tracker.process_order_fill(
                            symbol=managed_order.symbol,
                            quantity=quantity_change,
                            price=avg_fill_price,
                            venue="ALPACA",
                            order_id=managed_order.order_id
                        )
            
        except Exception as e:
            logger.error("Failed to sync order with manager",
                        alpaca_order_id=alpaca_order.id,
                        error=str(e))
    
    def _map_alpaca_order_type(self, alpaca_type: str) -> OrderType:
        """Map Alpaca order type to internal order type"""
        mapping = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop": OrderType.STOP,
            "stop_limit": OrderType.STOP_LIMIT,
            "trailing_stop": OrderType.TRAILING_STOP
        }
        return mapping.get(alpaca_type, OrderType.MARKET)
    
    def _map_alpaca_time_in_force(self, alpaca_tif: str) -> TimeInForce:
        """Map Alpaca time in force to internal time in force"""
        mapping = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
            "fok": TimeInForce.FOK
        }
        return mapping.get(alpaca_tif, TimeInForce.DAY)
    
    def _map_alpaca_order_status(self, alpaca_status: str) -> OrderStatus:
        """Map Alpaca order status to internal order status"""
        mapping = {
            "new": OrderStatus.SUBMITTED,
            "accepted": OrderStatus.ACCEPTED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "done_for_day": OrderStatus.CANCELLED,
            "canceled": OrderStatus.CANCELLED,
            "expired": OrderStatus.EXPIRED,
            "replaced": OrderStatus.ACCEPTED,
            "pending_cancel": OrderStatus.ACCEPTED,
            "pending_replace": OrderStatus.ACCEPTED,
            "rejected": OrderStatus.REJECTED,
            "suspended": OrderStatus.SUSPENDED
        }
        return mapping.get(alpaca_status, OrderStatus.PENDING)
    
    # === MONITORING AND CALLBACKS ===
    
    async def start_monitoring(self) -> None:
        """Start monitoring services"""
        logger.info("Starting broker monitoring services")
        
        tasks = []
        
        if self.order_manager:
            tasks.append(self.order_manager.start_monitoring())
        
        if self.position_tracker:
            tasks.append(self.position_tracker.start_monitoring())
        
        if self.account_monitor:
            tasks.append(self.account_monitor.start_monitoring())
        
        if tasks:
            await asyncio.gather(*tasks)
        
        logger.info("Broker monitoring services started")
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring services"""
        logger.info("Stopping broker monitoring services")
        
        tasks = []
        
        if self.order_manager:
            tasks.append(self.order_manager.stop_monitoring())
        
        if self.position_tracker:
            tasks.append(self.position_tracker.stop_monitoring())
        
        if self.account_monitor:
            tasks.append(self.account_monitor.stop_monitoring())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Broker monitoring services stopped")
    
    def get_broker_status(self) -> Dict[str, Any]:
        """Get comprehensive broker status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'connected': self.is_connected,
            'paper_trading': self.paper_trading,
            'market_open': self.is_market_open() if self.is_connected else None,
            'components': {
                'order_manager': self.order_manager is not None,
                'position_tracker': self.position_tracker is not None,
                'account_monitor': self.account_monitor is not None
            }
        }
        
        # Add component-specific status
        if self.order_manager:
            status['order_summary'] = self.order_manager.get_orders_summary()
        
        if self.position_tracker:
            status['portfolio_summary'] = self.position_tracker.get_portfolio_summary()
        
        if self.account_monitor:
            status['account_status'] = self.account_monitor.get_current_status()
        
        return status