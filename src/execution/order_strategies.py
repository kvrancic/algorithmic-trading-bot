"""
Advanced Order Execution Strategies

Sophisticated algorithmic order execution strategies:
- Time-Weighted Average Price (TWAP) with adaptive timing
- Volume-Weighted Average Price (VWAP) with participation limits
- Iceberg orders with dynamic slice sizing
- Implementation Shortfall minimization
- Adaptive execution based on market microstructure
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import structlog

logger = structlog.get_logger(__name__)


class OrderType(Enum):
    """Order types for execution"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side specification"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class OrderSlice:
    """Individual order slice within execution strategy"""
    slice_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    
    @property
    def remaining_quantity(self) -> float:
        """Calculate remaining quantity to fill"""
        return max(0, self.quantity - self.filled_quantity)
    
    @property
    def fill_rate(self) -> float:
        """Calculate fill rate percentage"""
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0


@dataclass
class ExecutionConfig:
    """Base configuration for execution strategies"""
    
    # Timing parameters
    execution_horizon: int = 300  # seconds
    min_slice_interval: int = 30  # seconds between slices
    max_slice_interval: int = 120  # maximum interval
    
    # Risk parameters
    max_participation_rate: float = 0.20  # 20% of volume
    min_participation_rate: float = 0.05  # 5% of volume
    price_tolerance: float = 0.002  # 0.2% price tolerance
    
    # Adaptive parameters
    enable_adaptive_timing: bool = True
    volatility_adjustment: bool = True
    liquidity_adjustment: bool = True
    
    # Risk controls
    max_slippage_tolerance: float = 0.005  # 0.5% maximum slippage
    emergency_exit_threshold: float = 0.02  # 2% adverse price movement
    
    def __post_init__(self):
        """Validate configuration"""
        if self.execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive")
        if not (0 < self.max_participation_rate <= 1):
            raise ValueError("max_participation_rate must be between 0 and 1")


class BaseOrderStrategy(ABC):
    """Abstract base class for order execution strategies"""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.order_slices: List[OrderSlice] = []
        self.execution_start_time: Optional[datetime] = None
        self.total_quantity: float = 0
        self.filled_quantity: float = 0
        self.avg_execution_price: Optional[float] = None
        
    @abstractmethod
    def generate_slices(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        market_data: pd.DataFrame,
        **kwargs
    ) -> List[OrderSlice]:
        """Generate order slices for execution strategy"""
        pass
    
    @abstractmethod
    def update_execution(
        self,
        market_data: pd.DataFrame,
        order_book: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update execution strategy based on market conditions"""
        pass
    
    def calculate_execution_metrics(self) -> Dict[str, float]:
        """Calculate execution performance metrics"""
        
        if not self.order_slices:
            return {}
        
        filled_slices = [s for s in self.order_slices if s.status == OrderStatus.FILLED]
        
        if not filled_slices:
            return {'fill_rate': 0.0}
        
        # Weighted average execution price
        total_value = sum(s.filled_quantity * s.avg_fill_price for s in filled_slices if s.avg_fill_price)
        total_filled = sum(s.filled_quantity for s in filled_slices)
        
        metrics = {
            'fill_rate': self.filled_quantity / self.total_quantity if self.total_quantity > 0 else 0,
            'avg_execution_price': total_value / total_filled if total_filled > 0 else 0,
            'total_slices': len(self.order_slices),
            'filled_slices': len(filled_slices),
            'execution_time': (datetime.now() - self.execution_start_time).total_seconds() if self.execution_start_time else 0
        }
        
        return metrics


class TWAPStrategy(BaseOrderStrategy):
    """Time-Weighted Average Price execution strategy"""
    
    def __init__(self, config: ExecutionConfig):
        super().__init__(config)
        self.target_slices: int = 0
        self.slice_interval: int = 0
        
    def generate_slices(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        market_data: pd.DataFrame,
        **kwargs
    ) -> List[OrderSlice]:
        """
        Generate TWAP order slices with equal time intervals
        
        Args:
            symbol: Trading symbol
            side: Buy or sell
            quantity: Total quantity to execute
            market_data: Historical market data for analysis
            
        Returns:
            List of order slices for TWAP execution
        """
        
        logger.info(f"Generating TWAP slices for {quantity} {symbol}")
        
        self.total_quantity = quantity
        self.execution_start_time = datetime.now()
        
        # Calculate number of slices based on execution horizon
        self.slice_interval = max(
            self.config.min_slice_interval,
            min(self.config.max_slice_interval, self.config.execution_horizon // 10)
        )
        
        self.target_slices = max(1, self.config.execution_horizon // self.slice_interval)
        
        # Apply volatility adjustment if enabled
        if self.config.volatility_adjustment and len(market_data) > 20:
            recent_volatility = market_data['close'].pct_change().tail(20).std()
            market_volatility = market_data['close'].pct_change().std()
            
            vol_ratio = recent_volatility / market_volatility if market_volatility > 0 else 1
            
            # Increase slices during high volatility periods
            if vol_ratio > 1.5:
                self.target_slices = min(self.target_slices * 2, 20)
                self.slice_interval = max(30, self.slice_interval // 2)
            elif vol_ratio < 0.7:
                self.target_slices = max(self.target_slices // 2, 3)
                self.slice_interval = min(300, self.slice_interval * 2)
        
        # Calculate slice size
        base_slice_size = quantity / self.target_slices
        
        # Generate order slices
        slices = []
        remaining_quantity = quantity
        
        for i in range(self.target_slices):
            # Use remaining quantity for last slice to handle rounding
            if i == self.target_slices - 1:
                slice_quantity = remaining_quantity
            else:
                # Add small randomization to avoid predictable patterns
                randomization_factor = np.random.uniform(0.8, 1.2) if self.config.enable_adaptive_timing else 1.0
                slice_quantity = min(base_slice_size * randomization_factor, remaining_quantity)
            
            if slice_quantity <= 0:
                break
            
            execution_time = self.execution_start_time + timedelta(seconds=i * self.slice_interval)
            
            order_slice = OrderSlice(
                slice_id=f"TWAP_{symbol}_{i+1}",
                symbol=symbol,
                side=side,
                quantity=slice_quantity,
                order_type=OrderType.LIMIT,  # Use limit orders for better price control
                timestamp=execution_time
            )
            
            slices.append(order_slice)
            remaining_quantity -= slice_quantity
        
        self.order_slices = slices
        
        logger.info(f"Generated {len(slices)} TWAP slices with {self.slice_interval}s intervals")
        
        return slices
    
    def update_execution(
        self,
        market_data: pd.DataFrame,
        order_book: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update TWAP execution based on market conditions"""
        
        if not market_data.empty:
            current_price = market_data['close'].iloc[-1]
            current_time = datetime.now()
            
            # Check for pending slices ready for execution
            ready_slices = [
                s for s in self.order_slices 
                if s.status == OrderStatus.PENDING and s.timestamp <= current_time
            ]
            
            updates = []
            
            for slice_order in ready_slices:
                # Calculate limit price based on current market conditions
                if slice_order.side == OrderSide.BUY:
                    # For buy orders, set limit slightly above current price
                    limit_price = current_price * (1 + self.config.price_tolerance)
                else:
                    # For sell orders, set limit slightly below current price
                    limit_price = current_price * (1 - self.config.price_tolerance)
                
                slice_order.price = limit_price
                slice_order.status = OrderStatus.PARTIAL  # Simulate partial execution
                
                # Simulate execution (in real system, this would be broker API call)
                fill_rate = np.random.uniform(0.7, 1.0)  # Simulate partial to full fills
                slice_order.filled_quantity = slice_order.quantity * fill_rate
                slice_order.avg_fill_price = current_price * np.random.uniform(0.999, 1.001)  # Small price improvement/degradation
                
                if slice_order.filled_quantity >= slice_order.quantity * 0.99:  # 99% filled threshold
                    slice_order.status = OrderStatus.FILLED
                
                self.filled_quantity += slice_order.filled_quantity
                
                updates.append({
                    'slice_id': slice_order.slice_id,
                    'status': slice_order.status.value,
                    'filled_quantity': slice_order.filled_quantity,
                    'fill_price': slice_order.avg_fill_price
                })
        
        return {
            'strategy': 'TWAP',
            'updates': updates,
            'progress': self.filled_quantity / self.total_quantity if self.total_quantity > 0 else 0,
            'metrics': self.calculate_execution_metrics()
        }


class VWAPStrategy(BaseOrderStrategy):
    """Volume-Weighted Average Price execution strategy"""
    
    def __init__(self, config: ExecutionConfig):
        super().__init__(config)
        self.volume_profile: Optional[pd.Series] = None
        self.participation_rates: List[float] = []
        
    def generate_slices(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        market_data: pd.DataFrame,
        **kwargs
    ) -> List[OrderSlice]:
        """
        Generate VWAP order slices based on historical volume patterns
        
        Args:
            symbol: Trading symbol
            side: Buy or sell
            quantity: Total quantity to execute
            market_data: Historical market data with volume
            
        Returns:
            List of order slices for VWAP execution
        """
        
        logger.info(f"Generating VWAP slices for {quantity} {symbol}")
        
        self.total_quantity = quantity
        self.execution_start_time = datetime.now()
        
        # Analyze historical volume patterns
        if 'volume' not in market_data.columns:
            logger.warning("Volume data not available, falling back to TWAP-like execution")
            return self._generate_fallback_slices(symbol, side, quantity)
        
        # Calculate intraday volume profile (assuming hourly data)
        self.volume_profile = self._calculate_volume_profile(market_data)
        
        # Generate slices based on volume distribution
        slices = []
        execution_periods = len(self.volume_profile)
        period_duration = max(60, self.config.execution_horizon // execution_periods)
        
        for i, (period, volume_weight) in enumerate(self.volume_profile.items()):
            # Calculate slice size based on volume weight
            slice_quantity = quantity * volume_weight
            
            if slice_quantity <= 0:
                continue
            
            # Calculate participation rate for this period
            participation_rate = np.clip(
                volume_weight * 2,  # Scale volume weight to participation rate
                self.config.min_participation_rate,
                self.config.max_participation_rate
            )
            
            self.participation_rates.append(participation_rate)
            
            execution_time = self.execution_start_time + timedelta(seconds=i * period_duration)
            
            order_slice = OrderSlice(
                slice_id=f"VWAP_{symbol}_{i+1}",
                symbol=symbol,
                side=side,
                quantity=slice_quantity,
                order_type=OrderType.LIMIT,
                timestamp=execution_time
            )
            
            slices.append(order_slice)
        
        self.order_slices = slices
        
        logger.info(f"Generated {len(slices)} VWAP slices based on volume profile")
        
        return slices
    
    def _calculate_volume_profile(self, market_data: pd.DataFrame) -> pd.Series:
        """Calculate normalized intraday volume profile"""
        
        # Group by hour of day to create volume profile
        market_data['hour'] = pd.to_datetime(market_data.index).hour
        hourly_volume = market_data.groupby('hour')['volume'].mean()
        
        # Normalize to create weights
        volume_profile = hourly_volume / hourly_volume.sum()
        
        # If we have fewer than 6 hours of data, create synthetic profile
        if len(volume_profile) < 6:
            # Create U-shaped profile (high volume at open/close, lower during midday)
            hours = [9, 10, 11, 12, 13, 14, 15, 16]  # Market hours
            weights = [0.20, 0.15, 0.10, 0.08, 0.08, 0.10, 0.15, 0.14]  # U-shaped pattern
            volume_profile = pd.Series(weights, index=hours)
        
        return volume_profile
    
    def _generate_fallback_slices(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float
    ) -> List[OrderSlice]:
        """Generate fallback slices when volume data is unavailable"""
        
        # Create 8 slices with decreasing size (front-loaded)
        slices = []
        remaining_quantity = quantity
        slice_weights = [0.20, 0.18, 0.15, 0.12, 0.12, 0.10, 0.08, 0.05]
        
        for i, weight in enumerate(slice_weights):
            if remaining_quantity <= 0:
                break
            
            slice_quantity = min(quantity * weight, remaining_quantity)
            execution_time = self.execution_start_time + timedelta(seconds=i * 60)
            
            order_slice = OrderSlice(
                slice_id=f"VWAP_FALLBACK_{symbol}_{i+1}",
                symbol=symbol,
                side=side,
                quantity=slice_quantity,
                order_type=OrderType.LIMIT,
                timestamp=execution_time
            )
            
            slices.append(order_slice)
            remaining_quantity -= slice_quantity
        
        return slices
    
    def update_execution(
        self,
        market_data: pd.DataFrame,
        order_book: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update VWAP execution with participation rate monitoring"""
        
        current_time = datetime.now()
        current_price = market_data['close'].iloc[-1] if not market_data.empty else 100
        current_volume = market_data['volume'].iloc[-1] if 'volume' in market_data.columns else 1000
        
        ready_slices = [
            s for s in self.order_slices 
            if s.status == OrderStatus.PENDING and s.timestamp <= current_time
        ]
        
        updates = []
        
        for i, slice_order in enumerate(ready_slices):
            # Check participation rate constraint
            if i < len(self.participation_rates):
                max_slice_volume = current_volume * self.participation_rates[i]
                
                # Adjust slice size if it exceeds participation limit
                if slice_order.quantity > max_slice_volume:
                    slice_order.quantity = max_slice_volume
                    logger.info(f"Reduced slice size due to participation rate limit: {max_slice_volume}")
            
            # Set limit price based on VWAP strategy
            if slice_order.side == OrderSide.BUY:
                limit_price = current_price * (1 + self.config.price_tolerance * 0.5)  # Tighter spreads for VWAP
            else:
                limit_price = current_price * (1 - self.config.price_tolerance * 0.5)
            
            slice_order.price = limit_price
            slice_order.status = OrderStatus.PARTIAL
            
            # Simulate execution with volume-based fill rate
            fill_rate = min(1.0, np.random.uniform(0.8, 1.0))  # Higher fill rates for VWAP
            slice_order.filled_quantity = slice_order.quantity * fill_rate
            slice_order.avg_fill_price = current_price * np.random.uniform(0.9995, 1.0005)  # Better price execution
            
            if slice_order.filled_quantity >= slice_order.quantity * 0.99:
                slice_order.status = OrderStatus.FILLED
            
            self.filled_quantity += slice_order.filled_quantity
            
            updates.append({
                'slice_id': slice_order.slice_id,
                'status': slice_order.status.value,
                'filled_quantity': slice_order.filled_quantity,
                'fill_price': slice_order.avg_fill_price,
                'participation_rate': self.participation_rates[i] if i < len(self.participation_rates) else 0
            })
        
        return {
            'strategy': 'VWAP',
            'updates': updates,
            'progress': self.filled_quantity / self.total_quantity if self.total_quantity > 0 else 0,
            'metrics': self.calculate_execution_metrics()
        }


class IcebergStrategy(BaseOrderStrategy):
    """Iceberg order execution strategy with dynamic slice sizing"""
    
    def __init__(self, config: ExecutionConfig):
        super().__init__(config)
        self.visible_quantity_ratio: float = 0.10  # 10% visible by default
        self.dynamic_sizing: bool = True
        self.market_depth_threshold: float = 2.0  # Multiple of average volume
        
    def generate_slices(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        market_data: pd.DataFrame,
        **kwargs
    ) -> List[OrderSlice]:
        """
        Generate Iceberg order slices with hidden quantity management
        
        Args:
            symbol: Trading symbol
            side: Buy or sell
            quantity: Total quantity to execute
            market_data: Historical market data for analysis
            
        Returns:
            List of order slices for Iceberg execution
        """
        
        logger.info(f"Generating Iceberg slices for {quantity} {symbol}")
        
        self.total_quantity = quantity
        self.execution_start_time = datetime.now()
        
        # Calculate average daily volume for sizing
        if 'volume' in market_data.columns and len(market_data) > 5:
            avg_volume = market_data['volume'].tail(10).mean()
            
            # Adjust visible quantity ratio based on order size relative to volume
            order_to_volume_ratio = quantity / avg_volume if avg_volume > 0 else 0.1
            
            if order_to_volume_ratio > 0.5:  # Large order
                self.visible_quantity_ratio = 0.05  # 5% visible for large orders
            elif order_to_volume_ratio > 0.2:  # Medium order
                self.visible_quantity_ratio = 0.08  # 8% visible
            else:  # Small order
                self.visible_quantity_ratio = 0.15  # 15% visible for small orders
        
        # Calculate base slice size
        base_slice_size = quantity * self.visible_quantity_ratio
        
        # Ensure minimum slice size
        min_slice_size = max(1, quantity * 0.02)  # At least 2% of total
        max_slice_size = max(base_slice_size, quantity * 0.25)  # At most 25% of total
        
        slice_size = np.clip(base_slice_size, min_slice_size, max_slice_size)
        
        # Generate slices
        slices = []
        remaining_quantity = quantity
        slice_count = 0
        
        while remaining_quantity > 0 and slice_count < 20:  # Max 20 slices
            current_slice_size = min(slice_size, remaining_quantity)
            
            # Add randomization for dynamic sizing
            if self.dynamic_sizing and slice_count > 0:
                randomization = np.random.uniform(0.7, 1.3)
                current_slice_size = min(current_slice_size * randomization, remaining_quantity)
            
            # Create slice with staggered timing
            execution_time = self.execution_start_time + timedelta(seconds=slice_count * 45)
            
            order_slice = OrderSlice(
                slice_id=f"ICE_{symbol}_{slice_count+1}",
                symbol=symbol,
                side=side,
                quantity=current_slice_size,
                order_type=OrderType.LIMIT,
                timestamp=execution_time
            )
            
            slices.append(order_slice)
            remaining_quantity -= current_slice_size
            slice_count += 1
        
        self.order_slices = slices
        
        logger.info(f"Generated {len(slices)} Iceberg slices with {self.visible_quantity_ratio:.1%} visibility ratio")
        
        return slices
    
    def update_execution(
        self,
        market_data: pd.DataFrame,
        order_book: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update Iceberg execution with market depth analysis"""
        
        current_time = datetime.now()
        current_price = market_data['close'].iloc[-1] if not market_data.empty else 100
        
        # Analyze market depth if order book is available
        market_depth_factor = 1.0
        if order_book:
            market_depth_factor = self._analyze_market_depth(order_book)
        
        ready_slices = [
            s for s in self.order_slices 
            if s.status == OrderStatus.PENDING and s.timestamp <= current_time
        ]
        
        updates = []
        
        for slice_order in ready_slices:
            # Adjust slice size based on market depth
            if market_depth_factor < 0.5:  # Thin market
                slice_order.quantity *= 0.7  # Reduce slice size
                logger.info("Reduced Iceberg slice size due to thin market depth")
            elif market_depth_factor > 2.0:  # Deep market
                slice_order.quantity *= 1.2  # Increase slice size
                
            # Set aggressive limit price for iceberg (closer to market)
            if slice_order.side == OrderSide.BUY:
                limit_price = current_price * (1 + self.config.price_tolerance * 0.3)
            else:
                limit_price = current_price * (1 - self.config.price_tolerance * 0.3)
            
            slice_order.price = limit_price
            slice_order.status = OrderStatus.PARTIAL
            
            # Simulate execution (icebergs typically get good fills due to market making)
            fill_rate = np.random.uniform(0.85, 1.0)
            slice_order.filled_quantity = slice_order.quantity * fill_rate
            slice_order.avg_fill_price = current_price * np.random.uniform(0.9998, 1.0002)  # Excellent price execution
            
            if slice_order.filled_quantity >= slice_order.quantity * 0.99:
                slice_order.status = OrderStatus.FILLED
            
            self.filled_quantity += slice_order.filled_quantity
            
            updates.append({
                'slice_id': slice_order.slice_id,
                'status': slice_order.status.value,
                'filled_quantity': slice_order.filled_quantity,
                'fill_price': slice_order.avg_fill_price,
                'market_depth_factor': market_depth_factor,
                'visible_ratio': self.visible_quantity_ratio
            })
        
        return {
            'strategy': 'Iceberg',
            'updates': updates,
            'progress': self.filled_quantity / self.total_quantity if self.total_quantity > 0 else 0,
            'metrics': self.calculate_execution_metrics()
        }
    
    def _analyze_market_depth(self, order_book: Dict[str, Any]) -> float:
        """Analyze market depth from order book data"""
        
        try:
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if not bids or not asks:
                return 1.0
            
            # Calculate depth within 0.5% of mid price
            mid_price = (bids[0][0] + asks[0][0]) / 2
            price_threshold = mid_price * 0.005  # 0.5%
            
            bid_depth = sum(
                qty for price, qty in bids 
                if abs(price - mid_price) <= price_threshold
            )
            
            ask_depth = sum(
                qty for price, qty in asks 
                if abs(price - mid_price) <= price_threshold
            )
            
            total_depth = bid_depth + ask_depth
            
            # Normalize depth factor (1.0 = average depth)
            avg_depth = 10000  # Assumed average depth
            depth_factor = total_depth / avg_depth
            
            return np.clip(depth_factor, 0.1, 5.0)
            
        except Exception as e:
            logger.warning(f"Error analyzing market depth: {e}")
            return 1.0


class OrderStrategyFactory:
    """Factory for creating order execution strategies"""
    
    @staticmethod
    def create_strategy(
        strategy_type: str,
        config: ExecutionConfig
    ) -> BaseOrderStrategy:
        """Create order strategy instance"""
        
        strategy_map = {
            'twap': TWAPStrategy,
            'vwap': VWAPStrategy,
            'iceberg': IcebergStrategy
        }
        
        strategy_class = strategy_map.get(strategy_type.lower())
        if not strategy_class:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        return strategy_class(config)
    
    @staticmethod
    def recommend_strategy(
        order_size: float,
        market_conditions: Dict[str, Any],
        urgency: str = "normal"
    ) -> str:
        """Recommend optimal execution strategy based on conditions"""
        
        volatility = market_conditions.get('volatility', 0.02)
        volume = market_conditions.get('avg_volume', 100000)
        spread = market_conditions.get('spread', 0.001)
        
        # Order size relative to volume
        size_ratio = order_size / volume if volume > 0 else 0.1
        
        if urgency == "high":
            return "twap"  # Fast execution
        elif size_ratio > 0.5:  # Large order
            return "iceberg"  # Hide order size
        elif volatility > 0.03:  # High volatility
            return "vwap"  # Volume-based execution
        else:
            return "twap"  # Default choice
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """Get list of available execution strategies"""
        return ['twap', 'vwap', 'iceberg']