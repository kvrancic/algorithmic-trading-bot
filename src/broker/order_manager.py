"""
Order Management System

Comprehensive order lifecycle management with:
- Order creation, modification, and cancellation
- Status tracking and event handling
- Integration with smart order router
- Risk controls and validation
- Real-time order monitoring
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import asyncio
from collections import defaultdict
import structlog

logger = structlog.get_logger(__name__)


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    SUSPENDED = "suspended"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    MOO = "market_on_open"
    MOC = "market_on_close"


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"


class TimeInForce(Enum):
    """Time in force enumeration"""
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


@dataclass
class OrderFill:
    """Individual order fill record"""
    fill_id: str
    order_id: str
    quantity: float
    price: float
    timestamp: datetime
    venue: str
    commission: float = 0.0
    
    @property
    def notional_value(self) -> float:
        """Calculate notional value of fill"""
        return self.quantity * self.price


@dataclass
class Order:
    """Comprehensive order representation"""
    
    # Basic order information
    order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    time_in_force: TimeInForce = TimeInForce.DAY
    
    # Pricing
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_amount: Optional[float] = None
    trail_percent: Optional[float] = None
    
    # Status and timing
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    
    # Execution details
    filled_quantity: float = 0.0
    remaining_quantity: Optional[float] = None
    average_fill_price: Optional[float] = None
    fills: List[OrderFill] = field(default_factory=list)
    
    # Venue and routing
    venue: Optional[str] = None
    routing_strategy: Optional[str] = None
    
    # Order metadata
    tags: Dict[str, Any] = field(default_factory=dict)
    parent_order_id: Optional[str] = None
    child_order_ids: List[str] = field(default_factory=list)
    
    # Rejection/error information
    rejection_reason: Optional[str] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Initialize derived fields"""
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity
    
    @property
    def is_buy(self) -> bool:
        """Check if order is a buy order"""
        return self.side == OrderSide.BUY
    
    @property
    def is_sell(self) -> bool:
        """Check if order is a sell order"""
        return self.side == OrderSide.SELL
    
    @property
    def is_active(self) -> bool:
        """Check if order is in active state"""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED
        ]
    
    @property
    def is_terminal(self) -> bool:
        """Check if order is in terminal state"""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        ]
    
    @property
    def fill_rate(self) -> float:
        """Calculate fill rate"""
        if self.quantity == 0:
            return 0.0
        return self.filled_quantity / self.quantity
    
    @property
    def total_commission(self) -> float:
        """Calculate total commission from all fills"""
        return sum(fill.commission for fill in self.fills)
    
    @property
    def notional_value(self) -> float:
        """Calculate notional value of order"""
        if self.average_fill_price:
            return self.filled_quantity * self.average_fill_price
        elif self.limit_price:
            return self.quantity * self.limit_price
        return 0.0
    
    def add_fill(self, fill: OrderFill) -> None:
        """Add a fill to the order"""
        self.fills.append(fill)
        self.filled_quantity += fill.quantity
        self.remaining_quantity = max(0, self.quantity - self.filled_quantity)
        
        # Update average fill price
        total_value = sum(f.quantity * f.price for f in self.fills)
        self.average_fill_price = total_value / self.filled_quantity if self.filled_quantity > 0 else None
        
        # Update status
        if self.remaining_quantity == 0:
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        self.last_updated = datetime.now()
        
        logger.info("Order fill added",
                   order_id=self.order_id,
                   fill_quantity=fill.quantity,
                   fill_price=fill.price,
                   total_filled=self.filled_quantity,
                   remaining=self.remaining_quantity)


@dataclass
class OrderManagerConfig:
    """Configuration for order manager"""
    
    # Order limits
    max_orders_per_symbol: int = 100
    max_total_orders: int = 1000
    order_timeout_minutes: int = 60
    
    # Risk controls
    max_order_value: float = 1000000.0  # $1M max per order
    max_position_concentration: float = 0.1  # 10% max per symbol
    enable_duplicate_prevention: bool = True
    
    # Monitoring
    enable_order_monitoring: bool = True
    monitoring_interval_seconds: int = 5
    stale_order_threshold_minutes: int = 15
    
    # Integration
    enable_smart_routing: bool = True
    default_venue: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration"""
        if self.max_orders_per_symbol <= 0:
            raise ValueError("max_orders_per_symbol must be positive")
        if self.max_total_orders <= 0:
            raise ValueError("max_total_orders must be positive")


class OrderManager:
    """Comprehensive order management system"""
    
    def __init__(self, config: OrderManagerConfig):
        self.config = config
        
        # Order storage
        self.orders: Dict[str, Order] = {}
        self.orders_by_symbol: Dict[str, List[str]] = defaultdict(list)
        self.orders_by_status: Dict[OrderStatus, List[str]] = defaultdict(list)
        
        # Event callbacks
        self.order_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.status_callbacks: Dict[OrderStatus, List[Callable]] = defaultdict(list)
        
        # Monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        # Integration components
        self.broker = None
        self.smart_router = None
        self.risk_engine = None
        
        logger.info("Order manager initialized", config=config)
    
    def set_broker(self, broker) -> None:
        """Set broker integration"""
        self.broker = broker
        logger.info("Broker integration set")
    
    def set_smart_router(self, smart_router) -> None:
        """Set smart order router"""
        self.smart_router = smart_router
        logger.info("Smart router integration set")
    
    def set_risk_engine(self, risk_engine) -> None:
        """Set risk engine"""
        self.risk_engine = risk_engine
        logger.info("Risk engine integration set")
    
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        client_order_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        routing_strategy: Optional[str] = None
    ) -> Order:
        """
        Create a new order with validation and risk checks
        
        Args:
            symbol: Trading symbol
            side: Buy or sell
            quantity: Order quantity
            order_type: Type of order
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: Time in force
            client_order_id: Custom order ID
            tags: Additional metadata
            routing_strategy: Execution strategy
            
        Returns:
            Created order
        """
        
        # Generate order ID
        order_id = str(uuid.uuid4())
        
        # Validate order
        await self._validate_order(symbol, side, quantity, order_type, limit_price, stop_price)
        
        # Create order object
        order = Order(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            tags=tags or {},
            routing_strategy=routing_strategy
        )
        
        # Store order
        self._store_order(order)
        
        # Trigger callbacks
        await self._trigger_order_event(order, "created")
        
        logger.info("Order created",
                   order_id=order_id,
                   symbol=symbol,
                   side=side.value,
                   quantity=quantity,
                   order_type=order_type.value)
        
        return order
    
    async def submit_order(self, order_id: str) -> bool:
        """
        Submit order to broker
        
        Args:
            order_id: Order to submit
            
        Returns:
            True if submission successful
        """
        
        order = self.orders.get(order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")
        
        if order.status != OrderStatus.PENDING:
            raise ValueError(f"Order {order_id} is not in pending status")
        
        try:
            # Update status
            await self._update_order_status(order, OrderStatus.SUBMITTED)
            
            # Use smart router if available and enabled
            if self.config.enable_smart_routing and self.smart_router:
                success = await self._submit_via_smart_router(order)
            else:
                success = await self._submit_via_broker(order)
            
            if success:
                order.submitted_at = datetime.now()
                await self._update_order_status(order, OrderStatus.ACCEPTED)
                logger.info("Order submitted successfully", order_id=order_id)
            else:
                await self._update_order_status(order, OrderStatus.REJECTED)
                logger.error("Order submission failed", order_id=order_id)
            
            return success
            
        except Exception as e:
            order.error_message = str(e)
            await self._update_order_status(order, OrderStatus.REJECTED)
            logger.error("Order submission error", order_id=order_id, error=str(e))
            return False
    
    async def cancel_order(self, order_id: str, reason: str = "user_requested") -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order to cancel
            reason: Cancellation reason
            
        Returns:
            True if cancellation successful
        """
        
        order = self.orders.get(order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")
        
        if not order.is_active:
            logger.warning("Cannot cancel non-active order", 
                          order_id=order_id, 
                          status=order.status.value)
            return False
        
        try:
            # Cancel with broker
            if self.broker:
                success = await self.broker.cancel_order(order_id)
                if not success:
                    logger.error("Broker cancellation failed", order_id=order_id)
                    return False
            
            # Update status
            await self._update_order_status(order, OrderStatus.CANCELLED)
            order.tags['cancellation_reason'] = reason
            
            logger.info("Order cancelled", order_id=order_id, reason=reason)
            return True
            
        except Exception as e:
            logger.error("Order cancellation error", order_id=order_id, error=str(e))
            return False
    
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
        
        order = self.orders.get(order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")
        
        if not order.is_active:
            raise ValueError(f"Cannot modify non-active order {order_id}")
        
        try:
            # Store original values
            original_quantity = order.quantity
            original_limit = order.limit_price
            original_stop = order.stop_price
            
            # Update order
            if quantity is not None:
                order.quantity = quantity
                order.remaining_quantity = quantity - order.filled_quantity
            if limit_price is not None:
                order.limit_price = limit_price
            if stop_price is not None:
                order.stop_price = stop_price
            
            # Validate modified order
            await self._validate_order(
                order.symbol, order.side, order.quantity, 
                order.order_type, order.limit_price, order.stop_price
            )
            
            # Modify with broker
            if self.broker:
                success = await self.broker.modify_order(order_id, quantity, limit_price, stop_price)
                if not success:
                    # Revert changes
                    order.quantity = original_quantity
                    order.limit_price = original_limit
                    order.stop_price = original_stop
                    order.remaining_quantity = original_quantity - order.filled_quantity
                    return False
            
            order.last_updated = datetime.now()
            await self._trigger_order_event(order, "modified")
            
            logger.info("Order modified", 
                       order_id=order_id,
                       quantity=quantity,
                       limit_price=limit_price,
                       stop_price=stop_price)
            return True
            
        except Exception as e:
            logger.error("Order modification error", order_id=order_id, error=str(e))
            return False
    
    async def process_fill(
        self,
        order_id: str,
        fill_quantity: float,
        fill_price: float,
        venue: str,
        timestamp: Optional[datetime] = None,
        commission: float = 0.0
    ) -> None:
        """
        Process an order fill
        
        Args:
            order_id: Order that was filled
            fill_quantity: Quantity filled
            fill_price: Fill price
            venue: Execution venue
            timestamp: Fill timestamp
            commission: Commission charged
        """
        
        order = self.orders.get(order_id)
        if not order:
            logger.error("Fill for unknown order", order_id=order_id)
            return
        
        # Create fill record
        fill = OrderFill(
            fill_id=str(uuid.uuid4()),
            order_id=order_id,
            quantity=fill_quantity,
            price=fill_price,
            timestamp=timestamp or datetime.now(),
            venue=venue,
            commission=commission
        )
        
        # Add fill to order
        order.add_fill(fill)
        
        # Update order tracking
        self._update_order_tracking(order)
        
        # Trigger callbacks
        await self._trigger_order_event(order, "filled")
        
        logger.info("Order fill processed",
                   order_id=order_id,
                   fill_quantity=fill_quantity,
                   fill_price=fill_price,
                   total_filled=order.filled_quantity,
                   remaining=order.remaining_quantity)
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol"""
        order_ids = self.orders_by_symbol.get(symbol, [])
        return [self.orders[oid] for oid in order_ids if oid in self.orders]
    
    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """Get all orders with specific status"""
        order_ids = self.orders_by_status.get(status, [])
        return [self.orders[oid] for oid in order_ids if oid in self.orders]
    
    def get_active_orders(self) -> List[Order]:
        """Get all active orders"""
        return [order for order in self.orders.values() if order.is_active]
    
    def get_orders_summary(self) -> Dict[str, Any]:
        """Get summary of all orders"""
        return {
            'total_orders': len(self.orders),
            'active_orders': len(self.get_active_orders()),
            'by_status': {
                status.value: len(self.get_orders_by_status(status))
                for status in OrderStatus
            },
            'by_symbol': {
                symbol: len(orders)
                for symbol, orders in self.orders_by_symbol.items()
            }
        }
    
    def register_order_callback(self, order_id: str, callback: Callable) -> None:
        """Register callback for specific order events"""
        self.order_callbacks[order_id].append(callback)
    
    def register_status_callback(self, status: OrderStatus, callback: Callable) -> None:
        """Register callback for status change events"""
        self.status_callbacks[status].append(callback)
    
    async def start_monitoring(self) -> None:
        """Start order monitoring task"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Order monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop order monitoring task"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Order monitoring stopped")
    
    async def _validate_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType,
        limit_price: Optional[float],
        stop_price: Optional[float]
    ) -> None:
        """Validate order parameters"""
        
        # Basic validation
        if quantity <= 0:
            raise ValueError("Order quantity must be positive")
        
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and limit_price is None:
            raise ValueError("Limit price required for limit orders")
        
        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and stop_price is None:
            raise ValueError("Stop price required for stop orders")
        
        # Order limits
        symbol_orders = len(self.get_orders_by_symbol(symbol))
        if symbol_orders >= self.config.max_orders_per_symbol:
            raise ValueError(f"Too many orders for symbol {symbol}")
        
        if len(self.orders) >= self.config.max_total_orders:
            raise ValueError("Maximum total orders exceeded")
        
        # Risk checks
        if self.risk_engine:
            risk_check = await self.risk_engine.validate_order(symbol, side, quantity, limit_price)
            if not risk_check.approved:
                raise ValueError(f"Risk check failed: {risk_check.reason}")
        
        # Order value check
        estimated_value = quantity * (limit_price or 100)  # Use limit price or estimate
        if estimated_value > self.config.max_order_value:
            raise ValueError(f"Order value {estimated_value} exceeds maximum {self.config.max_order_value}")
    
    def _store_order(self, order: Order) -> None:
        """Store order in internal tracking structures"""
        self.orders[order.order_id] = order
        self.orders_by_symbol[order.symbol].append(order.order_id)
        self.orders_by_status[order.status].append(order.order_id)
    
    async def _update_order_status(self, order: Order, new_status: OrderStatus) -> None:
        """Update order status and tracking"""
        old_status = order.status
        
        # Remove from old status tracking
        if order.order_id in self.orders_by_status[old_status]:
            self.orders_by_status[old_status].remove(order.order_id)
        
        # Update status
        order.status = new_status
        order.last_updated = datetime.now()
        
        # Add to new status tracking
        self.orders_by_status[new_status].append(order.order_id)
        
        # Trigger status callbacks
        await self._trigger_status_event(order, old_status, new_status)
    
    def _update_order_tracking(self, order: Order) -> None:
        """Update order in tracking structures"""
        # Order already stored, just ensure consistency
        if order.order_id not in self.orders:
            self._store_order(order)
    
    async def _submit_via_smart_router(self, order: Order) -> bool:
        """Submit order via smart router"""
        try:
            # Create execution plan
            execution_plan = await self.smart_router.create_execution_plan(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
                urgency="normal"  # Map from order priority/tags
            )
            
            # Execute order
            execution_results = await self.smart_router.execute_order(execution_plan)
            
            # Process results
            if execution_results.get('status') != 'failed':
                order.tags['execution_plan'] = execution_plan
                order.tags['routing_results'] = execution_results
                return True
            else:
                order.error_message = execution_results.get('error', 'Smart router execution failed')
                return False
                
        except Exception as e:
            logger.error("Smart router submission failed", order_id=order.order_id, error=str(e))
            order.error_message = str(e)
            return False
    
    async def _submit_via_broker(self, order: Order) -> bool:
        """Submit order directly to broker"""
        if not self.broker:
            logger.error("No broker configured for order submission")
            return False
        
        try:
            broker_order = await self.broker.submit_order(
                symbol=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
                order_type=order.order_type.value,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                time_in_force=order.time_in_force.value,
                client_order_id=order.client_order_id
            )
            
            # Store broker order reference
            order.tags['broker_order_id'] = broker_order.id if hasattr(broker_order, 'id') else None
            return True
            
        except Exception as e:
            logger.error("Broker submission failed", order_id=order.order_id, error=str(e))
            order.error_message = str(e)
            return False
    
    async def _trigger_order_event(self, order: Order, event_type: str) -> None:
        """Trigger order-specific callbacks"""
        callbacks = self.order_callbacks.get(order.order_id, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(order, event_type)
                else:
                    callback(order, event_type)
            except Exception as e:
                logger.error("Order callback error", 
                           order_id=order.order_id,
                           event_type=event_type,
                           error=str(e))
    
    async def _trigger_status_event(self, order: Order, old_status: OrderStatus, new_status: OrderStatus) -> None:
        """Trigger status change callbacks"""
        callbacks = self.status_callbacks.get(new_status, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(order, old_status, new_status)
                else:
                    callback(order, old_status, new_status)
            except Exception as e:
                logger.error("Status callback error",
                           order_id=order.order_id,
                           old_status=old_status.value,
                           new_status=new_status.value,
                           error=str(e))
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                await self._check_stale_orders()
                await self._check_order_timeouts()
                await self._sync_with_broker()
                
                await asyncio.sleep(self.config.monitoring_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Monitoring loop error", error=str(e))
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _check_stale_orders(self) -> None:
        """Check for stale orders that need status updates"""
        cutoff = datetime.now() - timedelta(minutes=self.config.stale_order_threshold_minutes)
        
        for order in self.get_active_orders():
            if order.last_updated and order.last_updated < cutoff:
                logger.warning("Stale order detected", 
                             order_id=order.order_id,
                             last_updated=order.last_updated)
                # Could trigger status refresh from broker
    
    async def _check_order_timeouts(self) -> None:
        """Check for expired orders"""
        cutoff = datetime.now() - timedelta(minutes=self.config.order_timeout_minutes)
        
        for order in self.get_active_orders():
            if order.created_at < cutoff:
                logger.warning("Order timeout detected", 
                             order_id=order.order_id,
                             created_at=order.created_at)
                await self._update_order_status(order, OrderStatus.EXPIRED)
    
    async def _sync_with_broker(self) -> None:
        """Synchronize order status with broker"""
        if not self.broker:
            return
        
        # Get active orders from broker and sync status
        try:
            broker_orders = await self.broker.get_orders(status='open')
            broker_order_ids = {order.client_order_id or order.id for order in broker_orders}
            
            # Check for orders that are no longer active at broker
            for order in self.get_active_orders():
                broker_ref = order.tags.get('broker_order_id') or order.client_order_id
                if broker_ref and broker_ref not in broker_order_ids:
                    # Order may have been filled or cancelled externally
                    logger.info("Order no longer active at broker", 
                              order_id=order.order_id,
                              broker_ref=broker_ref)
                    # Could trigger status refresh
                    
        except Exception as e:
            logger.error("Broker sync error", error=str(e))