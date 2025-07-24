"""
Position Tracking System

Comprehensive position management with:
- Real-time position tracking and P&L calculation
- Risk metrics and exposure monitoring
- Position lifecycle management
- Integration with order management
- Multi-venue position aggregation
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict
import structlog

logger = structlog.get_logger(__name__)


class PositionSide(Enum):
    """Position side enumeration"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class PositionUpdate:
    """Position update record"""
    update_id: str
    position_id: str
    symbol: str
    quantity_change: float
    price: float
    timestamp: datetime
    source: str  # order_fill, adjustment, etc.
    venue: Optional[str] = None
    commission: float = 0.0
    tags: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def notional_change(self) -> float:
        """Calculate notional value change"""
        return self.quantity_change * self.price


@dataclass
class Position:
    """Comprehensive position representation"""
    
    # Basic position info
    position_id: str
    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0
    
    # Cost basis and P&L
    cost_basis: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    
    # Position metadata
    opened_at: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)
    venue_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Tracking data
    updates: List[PositionUpdate] = field(default_factory=list)
    high_water_mark: float = 0.0
    low_water_mark: float = 0.0
    max_quantity: float = 0.0
    
    # Tags and metadata
    tags: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def side(self) -> PositionSide:
        """Get position side"""
        if self.quantity > 0:
            return PositionSide.LONG
        elif self.quantity < 0:
            return PositionSide.SHORT
        else:
            return PositionSide.FLAT
    
    @property
    def is_flat(self) -> bool:
        """Check if position is flat"""
        return abs(self.quantity) < 1e-8
    
    @property
    def is_long(self) -> bool:
        """Check if position is long"""
        return self.quantity > 1e-8
    
    @property
    def is_short(self) -> bool:
        """Check if position is short"""
        return self.quantity < -1e-8
    
    @property
    def notional_value(self) -> float:
        """Current notional value of position"""
        return abs(self.quantity * self.average_price)
    
    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)"""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def pnl_percent(self) -> float:
        """P&L as percentage of cost basis"""
        if self.cost_basis == 0:
            return 0.0
        return (self.total_pnl / abs(self.cost_basis)) * 100
    
    def update_market_price(self, market_price: float) -> None:
        """Update position with current market price"""
        if self.is_flat:
            self.unrealized_pnl = 0.0
            return
        
        # Calculate unrealized P&L
        if self.is_long:
            self.unrealized_pnl = (market_price - self.average_price) * self.quantity
        else:  # short position
            self.unrealized_pnl = (self.average_price - market_price) * abs(self.quantity)
        
        # Update high/low water marks
        current_total_pnl = self.total_pnl
        self.high_water_mark = max(self.high_water_mark, current_total_pnl)
        self.low_water_mark = min(self.low_water_mark, current_total_pnl)
        
        self.last_updated = datetime.now()
        
        logger.debug("Position market price updated",
                    symbol=self.symbol,
                    market_price=market_price,
                    unrealized_pnl=self.unrealized_pnl,
                    total_pnl=current_total_pnl)
    
    def add_update(self, update: PositionUpdate) -> None:
        """Add position update and recalculate metrics"""
        self.updates.append(update)
        
        # Update venue breakdown
        if update.venue:
            if update.venue not in self.venue_breakdown:
                self.venue_breakdown[update.venue] = 0.0
            self.venue_breakdown[update.venue] += update.quantity_change
        
        # Handle position change
        if update.quantity_change != 0:
            self._update_position_metrics(update)
        
        # Update commission
        self.total_commission += update.commission
        
        # Update timing
        if self.opened_at is None and not self.is_flat:
            self.opened_at = update.timestamp
        self.last_updated = update.timestamp
        
        logger.info("Position update added",
                   symbol=self.symbol,
                   quantity_change=update.quantity_change,
                   new_quantity=self.quantity,
                   average_price=self.average_price)
    
    def _update_position_metrics(self, update: PositionUpdate) -> None:
        """Update position metrics after quantity change"""
        old_quantity = self.quantity
        new_quantity = old_quantity + update.quantity_change
        
        # Handle position flattening
        if abs(new_quantity) < 1e-8:
            # Position closed - realize P&L
            if not self.is_flat:
                closing_pnl = self._calculate_closing_pnl(update)
                self.realized_pnl += closing_pnl
            
            self.quantity = 0.0
            self.average_price = 0.0
            self.cost_basis = 0.0
            self.unrealized_pnl = 0.0
            return
        
        # Handle position opening
        if self.is_flat:
            self.quantity = new_quantity
            self.average_price = update.price
            self.cost_basis = abs(new_quantity * update.price)
            self.max_quantity = abs(new_quantity)
            return
        
        # Handle position increase (same direction)
        if (old_quantity > 0 and update.quantity_change > 0) or \
           (old_quantity < 0 and update.quantity_change < 0):
            
            # Update average price using weighted average
            total_cost = abs(old_quantity * self.average_price) + abs(update.quantity_change * update.price)
            total_quantity = abs(new_quantity)
            self.average_price = total_cost / total_quantity if total_quantity > 0 else 0.0
            
            self.quantity = new_quantity
            self.cost_basis = abs(new_quantity * self.average_price)
            self.max_quantity = max(self.max_quantity, abs(new_quantity))
            
        # Handle position decrease (opposite direction)
        else:
            # Realize P&L on the portion being closed
            closing_quantity = min(abs(old_quantity), abs(update.quantity_change))
            closing_pnl = self._calculate_partial_closing_pnl(update, closing_quantity)
            self.realized_pnl += closing_pnl
            
            self.quantity = new_quantity
            
            # If still have position, keep same average price
            if not self.is_flat:
                self.cost_basis = abs(new_quantity * self.average_price)
            else:
                self.average_price = 0.0
                self.cost_basis = 0.0
    
    def _calculate_closing_pnl(self, update: PositionUpdate) -> float:
        """Calculate P&L when closing entire position"""
        if self.is_long:
            return (update.price - self.average_price) * abs(self.quantity)
        else:  # short position
            return (self.average_price - update.price) * abs(self.quantity)
    
    def _calculate_partial_closing_pnl(self, update: PositionUpdate, closing_quantity: float) -> float:
        """Calculate P&L when partially closing position"""
        if self.is_long:
            return (update.price - self.average_price) * closing_quantity
        else:  # short position
            return (self.average_price - update.price) * closing_quantity
    
    def get_risk_metrics(self, market_price: float) -> Dict[str, float]:
        """Calculate position risk metrics"""
        self.update_market_price(market_price)
        
        return {
            'notional_value': self.notional_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.total_pnl,
            'pnl_percent': self.pnl_percent,
            'cost_basis': self.cost_basis,
            'max_drawdown': self.high_water_mark - self.total_pnl if self.high_water_mark > 0 else 0,
            'max_runup': self.total_pnl - self.low_water_mark if self.low_water_mark < 0 else self.total_pnl,
            'commission_ratio': (self.total_commission / self.cost_basis * 100) if self.cost_basis > 0 else 0
        }


@dataclass
class PositionTrackerConfig:
    """Configuration for position tracker"""
    
    # Position limits
    max_positions: int = 100
    max_concentration_per_symbol: float = 0.1  # 10% max per symbol
    max_sector_concentration: float = 0.3  # 30% max per sector
    
    # Risk monitoring
    enable_real_time_pnl: bool = True
    pnl_update_interval_seconds: int = 1
    max_drawdown_threshold: float = 0.05  # 5% max drawdown alert
    
    # Position management
    auto_flatten_on_loss: bool = False
    max_loss_threshold: float = 0.1  # 10% max loss per position
    enable_position_sizing: bool = True
    
    # Data retention
    keep_closed_positions_days: int = 30
    max_updates_per_position: int = 1000
    
    # Integration
    enable_market_data_updates: bool = True
    market_data_update_interval: int = 5  # seconds
    
    def __post_init__(self):
        """Validate configuration"""
        if self.max_positions <= 0:
            raise ValueError("max_positions must be positive")
        if not (0 < self.max_concentration_per_symbol <= 1):
            raise ValueError("max_concentration_per_symbol must be between 0 and 1")


class PositionTracker:
    """Comprehensive position tracking system"""
    
    def __init__(self, config: PositionTrackerConfig):
        self.config = config
        
        # Position storage
        self.positions: Dict[str, Position] = {}
        self.positions_by_symbol: Dict[str, str] = {}  # symbol -> position_id
        self.closed_positions: List[Position] = []
        
        # Market data cache
        self.market_prices: Dict[str, float] = {}
        self.last_price_update: Dict[str, datetime] = {}
        
        # Portfolio metrics
        self.portfolio_value: float = 0.0
        self.total_pnl: float = 0.0
        self.total_commission: float = 0.0
        
        # Integration components
        self.broker = None
        self.market_data_feed = None
        self.risk_engine = None
        
        # Monitoring
        self.monitoring_active = False
        self.pnl_update_task = None
        
        logger.info("Position tracker initialized", config=config)
    
    def set_broker(self, broker) -> None:
        """Set broker integration"""
        self.broker = broker
        logger.info("Broker integration set for position tracker")
    
    def set_market_data_feed(self, market_data_feed) -> None:
        """Set market data feed"""
        self.market_data_feed = market_data_feed
        logger.info("Market data feed set for position tracker")
    
    def set_risk_engine(self, risk_engine) -> None:
        """Set risk engine"""
        self.risk_engine = risk_engine
        logger.info("Risk engine set for position tracker")
    
    async def process_order_fill(
        self,
        symbol: str,
        quantity: float,
        price: float,
        venue: str,
        order_id: str,
        timestamp: Optional[datetime] = None,
        commission: float = 0.0
    ) -> None:
        """
        Process order fill and update position
        
        Args:
            symbol: Trading symbol
            quantity: Fill quantity (positive for buy, negative for sell)
            price: Fill price
            venue: Execution venue
            order_id: Source order ID
            timestamp: Fill timestamp
            commission: Commission charged
        """
        
        position = self.get_or_create_position(symbol)
        
        # Create position update
        update = PositionUpdate(
            update_id=f"{order_id}_{len(position.updates)}",
            position_id=position.position_id,
            symbol=symbol,
            quantity_change=quantity,
            price=price,
            timestamp=timestamp or datetime.now(),
            source="order_fill",
            venue=venue,
            commission=commission,
            tags={'order_id': order_id}
        )
        
        # Add update to position
        position.add_update(update)
        
        # Update portfolio metrics
        await self._update_portfolio_metrics()
        
        # Check risk limits
        if self.risk_engine:
            await self._check_position_risk(position)
        
        # Handle position closure
        if position.is_flat:
            await self._handle_position_closure(position)
        
        logger.info("Order fill processed for position",
                   symbol=symbol,
                   quantity=quantity,
                   price=price,
                   new_position_qty=position.quantity,
                   position_pnl=position.total_pnl)
    
    def get_or_create_position(self, symbol: str) -> Position:
        """Get existing position or create new one"""
        position_id = self.positions_by_symbol.get(symbol)
        
        if position_id and position_id in self.positions:
            return self.positions[position_id]
        
        # Create new position
        position_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        position = Position(
            position_id=position_id,
            symbol=symbol
        )
        
        self.positions[position_id] = position
        self.positions_by_symbol[symbol] = position_id
        
        logger.info("New position created", symbol=symbol, position_id=position_id)
        return position
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        position_id = self.positions_by_symbol.get(symbol)
        if position_id:
            return self.positions.get(position_id)
        return None
    
    def get_all_positions(self) -> List[Position]:
        """Get all active positions"""
        return list(self.positions.values())
    
    def get_long_positions(self) -> List[Position]:
        """Get all long positions"""
        return [pos for pos in self.positions.values() if pos.is_long]
    
    def get_short_positions(self) -> List[Position]:
        """Get all short positions"""
        return [pos for pos in self.positions.values() if pos.is_short]
    
    def get_positions_by_symbol(self, symbols: List[str]) -> Dict[str, Optional[Position]]:
        """Get positions for multiple symbols"""
        return {symbol: self.get_position(symbol) for symbol in symbols}
    
    async def update_market_price(self, symbol: str, price: float) -> None:
        """Update market price for symbol"""
        self.market_prices[symbol] = price
        self.last_price_update[symbol] = datetime.now()
        
        # Update position if exists
        position = self.get_position(symbol)
        if position and not position.is_flat:
            position.update_market_price(price)
            
            # Check for risk alerts
            if self.risk_engine:
                await self._check_position_risk(position)
    
    async def update_all_market_prices(self, price_data: Dict[str, float]) -> None:
        """Update market prices for multiple symbols"""
        for symbol, price in price_data.items():
            await self.update_market_price(symbol, price)
        
        # Update portfolio metrics
        await self._update_portfolio_metrics()
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        active_positions = [pos for pos in self.positions.values() if not pos.is_flat]
        
        # Calculate totals
        total_notional = sum(pos.notional_value for pos in active_positions)
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in active_positions)
        total_realized_pnl = sum(pos.realized_pnl for pos in active_positions)
        total_commission = sum(pos.total_commission for pos in active_positions)
        
        # Position breakdown
        long_positions = len([pos for pos in active_positions if pos.is_long])
        short_positions = len([pos for pos in active_positions if pos.is_short])
        
        # Risk metrics
        positions_at_loss = len([pos for pos in active_positions if pos.total_pnl < 0])
        max_position_loss = min([pos.total_pnl for pos in active_positions], default=0)
        max_position_gain = max([pos.total_pnl for pos in active_positions], default=0)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'active_positions': len(active_positions),
            'long_positions': long_positions,
            'short_positions': short_positions,
            'total_notional_value': total_notional,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': total_realized_pnl,
            'total_pnl': total_unrealized_pnl + total_realized_pnl,
            'total_commission': total_commission,
            'positions_at_loss': positions_at_loss,
            'win_rate': (len(active_positions) - positions_at_loss) / len(active_positions) * 100 if active_positions else 0,
            'max_position_loss': max_position_loss,
            'max_position_gain': max_position_gain,
            'portfolio_concentration': self._calculate_concentration()
        }
    
    def get_position_summary(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed summary for specific position"""
        position = self.get_position(symbol)
        if not position:
            return None
        
        current_price = self.market_prices.get(symbol, position.average_price)
        risk_metrics = position.get_risk_metrics(current_price)
        
        return {
            'position_id': position.position_id,
            'symbol': position.symbol,
            'side': position.side.value,
            'quantity': position.quantity,
            'average_price': position.average_price,
            'current_price': current_price,
            'cost_basis': position.cost_basis,
            'market_value': abs(position.quantity * current_price),
            'unrealized_pnl': position.unrealized_pnl,
            'realized_pnl': position.realized_pnl,
            'total_pnl': position.total_pnl,
            'pnl_percent': position.pnl_percent,
            'total_commission': position.total_commission,
            'venue_breakdown': position.venue_breakdown,
            'opened_at': position.opened_at.isoformat() if position.opened_at else None,
            'last_updated': position.last_updated.isoformat(),
            'update_count': len(position.updates),
            'risk_metrics': risk_metrics
        }
    
    async def flatten_position(self, symbol: str, reason: str = "manual") -> bool:
        """
        Flatten position by creating closing order
        
        Args:
            symbol: Symbol to flatten
            reason: Reason for flattening
            
        Returns:
            True if flattening initiated successfully
        """
        position = self.get_position(symbol)
        if not position or position.is_flat:
            logger.warning("No position to flatten", symbol=symbol)
            return False
        
        if not self.broker:
            logger.error("No broker configured for position flattening")
            return False
        
        try:
            # Determine order side (opposite of position)
            side = "sell" if position.is_long else "buy"
            quantity = abs(position.quantity)
            
            # Create market order to flatten
            order = await self.broker.submit_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type="market",
                client_order_id=f"flatten_{symbol}_{datetime.now().strftime('%H%M%S')}"
            )
            
            logger.info("Position flattening order submitted",
                       symbol=symbol,
                       side=side,
                       quantity=quantity,
                       reason=reason,
                       order_id=getattr(order, 'id', 'unknown'))
            
            return True
            
        except Exception as e:
            logger.error("Failed to flatten position",
                        symbol=symbol,
                        error=str(e))
            return False
    
    async def sync_positions_with_broker(self) -> None:
        """Synchronize positions with broker"""
        if not self.broker:
            logger.warning("No broker configured for position sync")
            return
        
        try:
            broker_positions = await self.broker.get_positions()
            
            for broker_pos in broker_positions:
                symbol = broker_pos.symbol
                broker_qty = float(broker_pos.qty)
                
                local_position = self.get_position(symbol)
                local_qty = local_position.quantity if local_position else 0.0
                
                # Check for discrepancies
                if abs(broker_qty - local_qty) > 1e-6:
                    logger.warning("Position discrepancy detected",
                                 symbol=symbol,
                                 broker_qty=broker_qty,
                                 local_qty=local_qty)
                    
                    # Could implement reconciliation logic here
                    
        except Exception as e:
            logger.error("Position sync failed", error=str(e))
    
    async def start_monitoring(self) -> None:
        """Start position monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        if self.config.enable_real_time_pnl:
            import asyncio
            self.pnl_update_task = asyncio.create_task(self._pnl_update_loop())
        
        logger.info("Position monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop position monitoring"""
        self.monitoring_active = False
        
        if self.pnl_update_task:
            self.pnl_update_task.cancel()
            try:
                await self.pnl_update_task
            except:
                pass
        
        logger.info("Position monitoring stopped")
    
    def _calculate_concentration(self) -> Dict[str, float]:
        """Calculate portfolio concentration metrics"""
        if not self.positions:
            return {}
        
        active_positions = [pos for pos in self.positions.values() if not pos.is_flat]
        if not active_positions:
            return {}
        
        total_notional = sum(pos.notional_value for pos in active_positions)
        if total_notional == 0:
            return {}
        
        # Symbol concentration
        symbol_concentrations = {
            pos.symbol: pos.notional_value / total_notional
            for pos in active_positions
        }
        
        # Find max concentration
        max_symbol_concentration = max(symbol_concentrations.values()) if symbol_concentrations else 0
        
        return {
            'max_symbol_concentration': max_symbol_concentration,
            'symbol_concentrations': symbol_concentrations,
            'diversification_ratio': len(active_positions) / len(self.positions) if self.positions else 0
        }
    
    async def _update_portfolio_metrics(self) -> None:
        """Update portfolio-level metrics"""
        active_positions = [pos for pos in self.positions.values() if not pos.is_flat]
        
        self.portfolio_value = sum(pos.notional_value for pos in active_positions)
        self.total_pnl = sum(pos.total_pnl for pos in active_positions)
        self.total_commission = sum(pos.total_commission for pos in active_positions)
    
    async def _check_position_risk(self, position: Position) -> None:
        """Check position risk limits"""
        if not position or position.is_flat:
            return
        
        current_price = self.market_prices.get(position.symbol, position.average_price)
        risk_metrics = position.get_risk_metrics(current_price)
        
        # Check max loss threshold
        if self.config.auto_flatten_on_loss and risk_metrics['pnl_percent'] < -self.config.max_loss_threshold * 100:
            logger.warning("Position loss threshold exceeded",
                          symbol=position.symbol,
                          pnl_percent=risk_metrics['pnl_percent'])
            
            await self.flatten_position(position.symbol, "risk_management")
        
        # Check drawdown threshold
        if risk_metrics['max_drawdown'] > self.config.max_drawdown_threshold * position.cost_basis:
            logger.warning("Position drawdown threshold exceeded",
                          symbol=position.symbol,
                          max_drawdown=risk_metrics['max_drawdown'])
    
    async def _handle_position_closure(self, position: Position) -> None:
        """Handle position closure"""
        logger.info("Position closed",
                   symbol=position.symbol,
                   realized_pnl=position.realized_pnl,
                   total_commission=position.total_commission)
        
        # Move to closed positions
        self.closed_positions.append(position)
        
        # Remove from active tracking
        if position.position_id in self.positions:
            del self.positions[position.position_id]
        
        if position.symbol in self.positions_by_symbol:
            del self.positions_by_symbol[position.symbol]
        
        # Clean up old closed positions
        cutoff_date = datetime.now() - timedelta(days=self.config.keep_closed_positions_days)
        self.closed_positions = [
            pos for pos in self.closed_positions
            if pos.last_updated > cutoff_date
        ]
    
    async def _pnl_update_loop(self) -> None:
        """Main P&L update loop"""
        import asyncio
        
        while self.monitoring_active:
            try:
                # Update market prices if market data feed available
                if self.market_data_feed:
                    symbols = list(self.positions_by_symbol.keys())
                    if symbols:
                        price_data = await self.market_data_feed.get_latest_prices(symbols)
                        await self.update_all_market_prices(price_data)
                
                await asyncio.sleep(self.config.pnl_update_interval_seconds)
                
            except Exception as e:
                logger.error("P&L update loop error", error=str(e))
                await asyncio.sleep(5)  # Brief pause on error