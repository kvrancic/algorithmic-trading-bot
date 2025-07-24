"""
Smart Order Routing System

Intelligent order routing and execution optimization:
- Integration of all execution components (strategies, slippage, impact, orderbook, tracking)
- Real-time routing decisions based on market conditions and venue performance
- Dynamic strategy selection and parameter optimization
- Cross-venue execution coordination and smart order fragmentation
- Comprehensive execution monitoring and adaptive learning
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import structlog

from .order_strategies import OrderStrategyFactory, BaseOrderStrategy, ExecutionConfig, OrderSide
from .slippage_predictor import SlippagePredictor, SlippagePredictorConfig
from .market_impact import MarketImpactMinimizer, MarketImpactConfig
from .orderbook_analyzer import OrderBookAnalyzer, OrderBookConfig, OrderBookSnapshot
from .execution_tracker import ExecutionTracker, ExecutionConfig as TrackerConfig

logger = structlog.get_logger(__name__)


class RoutingDecision(Enum):
    """Smart routing decision types"""
    SINGLE_VENUE = "single_venue"
    MULTI_VENUE = "multi_venue"
    FRAGMENTED = "fragmented"
    ICEBERG = "iceberg"
    DELAYED = "delayed"


@dataclass
class RoutingConfig:
    """Configuration for smart order routing"""
    
    # Strategy selection
    enable_dynamic_strategy_selection: bool = True
    strategy_selection_confidence_threshold: float = 0.7
    default_strategy: str = "twap"
    
    # Venue routing
    enable_multi_venue_routing: bool = True
    max_venues_per_order: int = 3
    min_venue_allocation: float = 0.1  # 10% minimum allocation
    venue_selection_method: str = "performance_based"  # "performance_based", "liquidity_based", "cost_based"
    
    # Order fragmentation
    enable_order_fragmentation: bool = True
    max_fragments_per_order: int = 5
    fragmentation_threshold: float = 0.5  # Fragment if order > 50% of venue capacity
    
    # Timing optimization
    enable_timing_optimization: bool = True
    market_impact_threshold: float = 0.002  # 20 bps threshold
    delay_optimization_window: int = 300  # seconds
    
    # Adaptive learning
    enable_adaptive_learning: bool = True
    learning_window: int = 100  # number of executions for learning
    performance_feedback_weight: float = 0.3
    
    # Risk controls
    max_execution_duration: int = 3600  # 1 hour maximum
    emergency_execution_threshold: float = 0.05  # 5% adverse movement triggers emergency
    
    # Integration settings
    slippage_prediction_enabled: bool = True
    market_impact_optimization_enabled: bool = True
    orderbook_analysis_enabled: bool = True
    execution_tracking_enabled: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        if not (0 < self.strategy_selection_confidence_threshold < 1):
            raise ValueError("strategy_selection_confidence_threshold must be between 0 and 1")
        if self.max_venues_per_order < 1:
            raise ValueError("max_venues_per_order must be positive")


@dataclass
class VenueInfo:
    """Information about execution venue"""
    name: str
    liquidity: float
    latency_ms: float
    cost_per_share: float
    success_rate: float
    avg_slippage_bps: float
    capacity: float
    market_hours: Tuple[int, int] = (9, 16)  # Market hours
    
    @property
    def is_available(self) -> bool:
        """Check if venue is currently available"""
        current_hour = datetime.now().hour
        return self.market_hours[0] <= current_hour < self.market_hours[1]


@dataclass
class ExecutionPlan:
    """Comprehensive execution plan"""
    order_id: str
    routing_decision: RoutingDecision
    selected_strategy: str
    venue_allocations: Dict[str, float]
    execution_schedule: List[Dict[str, Any]]
    expected_performance: Dict[str, float]
    risk_assessment: Dict[str, Any]
    contingency_plans: List[Dict[str, Any]]
    
    @property
    def total_expected_cost_bps(self) -> float:
        """Calculate total expected execution cost"""
        return (
            self.expected_performance.get('slippage_bps', 0) +
            self.expected_performance.get('market_impact_bps', 0) +
            self.expected_performance.get('timing_cost_bps', 0)
        )


class StrategySelector:
    """Select optimal execution strategy based on conditions"""
    
    def __init__(self, config: RoutingConfig):
        self.config = config
        self.strategy_performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
    def select_strategy(
        self,
        order_characteristics: Dict[str, Any],
        market_conditions: Dict[str, Any],
        venue_conditions: Dict[str, Any]
    ) -> Tuple[str, float]:
        """
        Select optimal execution strategy
        
        Args:
            order_characteristics: Order size, urgency, etc.
            market_conditions: Volatility, volume, spread, etc.
            venue_conditions: Available venues and their characteristics
            
        Returns:
            Tuple of (strategy_name, confidence_score)
        """
        
        if not self.config.enable_dynamic_strategy_selection:
            return self.config.default_strategy, 0.5
        
        # Strategy scoring based on conditions
        strategy_scores = {}
        
        # Extract key characteristics
        order_size = order_characteristics.get('size', 0)
        urgency = order_characteristics.get('urgency', 'normal')
        symbol = order_characteristics.get('symbol', '')
        
        volatility = market_conditions.get('volatility', 0.02)
        volume = market_conditions.get('volume', 100000)
        spread_bps = market_conditions.get('spread_bps', 10)
        
        # Score TWAP strategy
        twap_score = self._score_twap_strategy(
            order_size, urgency, volatility, volume, spread_bps
        )
        strategy_scores['twap'] = twap_score
        
        # Score VWAP strategy
        vwap_score = self._score_vwap_strategy(
            order_size, urgency, volatility, volume, spread_bps
        )
        strategy_scores['vwap'] = vwap_score
        
        # Score Iceberg strategy
        iceberg_score = self._score_iceberg_strategy(
            order_size, urgency, volatility, volume, spread_bps
        )
        strategy_scores['iceberg'] = iceberg_score
        
        # Apply historical performance adjustment
        if self.config.enable_adaptive_learning:
            strategy_scores = self._apply_historical_performance(
                strategy_scores, symbol, market_conditions
            )
        
        # Select best strategy
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        confidence = strategy_scores[best_strategy]
        
        # Check confidence threshold
        if confidence < self.config.strategy_selection_confidence_threshold:
            logger.warning(f"Low confidence in strategy selection: {confidence:.2f}")
            best_strategy = self.config.default_strategy
            confidence = 0.5
        
        logger.info(f"Selected strategy: {best_strategy} (confidence: {confidence:.2f})")
        
        return best_strategy, confidence
    
    def _score_twap_strategy(
        self,
        order_size: float,
        urgency: str,
        volatility: float,
        volume: float,
        spread_bps: float
    ) -> float:
        """Score TWAP strategy appropriateness"""
        
        score = 0.5  # Base score
        
        # TWAP is good for:
        # - Normal urgency orders
        # - Moderate volatility
        # - When order size is not too large relative to volume
        
        if urgency == 'normal':
            score += 0.2
        elif urgency == 'high':
            score -= 0.1
        elif urgency == 'low':
            score += 0.1
        
        # Volatility consideration
        if 0.15 <= volatility <= 0.25:  # Moderate volatility
            score += 0.2
        elif volatility > 0.35:  # High volatility
            score -= 0.1
        
        # Order size vs volume
        size_ratio = order_size / volume if volume > 0 else 0.1
        if size_ratio < 0.1:  # Small order
            score += 0.1
        elif size_ratio > 0.5:  # Large order
            score -= 0.2
        
        # Spread consideration
        if spread_bps < 15:  # Tight spread
            score += 0.1
        
        return np.clip(score, 0, 1)
    
    def _score_vwap_strategy(
        self,
        order_size: float,
        urgency: str,
        volatility: float,
        volume: float,
        spread_bps: float
    ) -> float:
        """Score VWAP strategy appropriateness"""
        
        score = 0.5  # Base score
        
        # VWAP is good for:
        # - Large orders
        # - High volume markets
        # - When you want to match market participation
        
        size_ratio = order_size / volume if volume > 0 else 0.1
        if size_ratio > 0.2:  # Large order
            score += 0.3
        elif size_ratio < 0.05:  # Small order
            score -= 0.2
        
        # Volume consideration
        if volume > 500000:  # High volume
            score += 0.2
        elif volume < 50000:  # Low volume
            score -= 0.3
        
        # Urgency
        if urgency == 'low':
            score += 0.1
        elif urgency == 'high':
            score -= 0.2
        
        # Volatility
        if volatility < 0.20:  # Lower volatility
            score += 0.1
        
        return np.clip(score, 0, 1)
    
    def _score_iceberg_strategy(
        self,
        order_size: float,
        urgency: str,
        volatility: float,
        volume: float,
        spread_bps: float
    ) -> float:
        """Score Iceberg strategy appropriateness"""
        
        score = 0.4  # Base score (slightly lower default)
        
        # Iceberg is good for:
        # - Very large orders
        # - When you want to hide order size
        # - High liquidity markets
        
        size_ratio = order_size / volume if volume > 0 else 0.1
        if size_ratio > 0.5:  # Very large order
            score += 0.4
        elif size_ratio > 0.3:  # Large order
            score += 0.2
        elif size_ratio < 0.1:  # Small order
            score -= 0.3
        
        # Market depth consideration
        if volume > 1000000:  # Very high volume (good for iceberg)
            score += 0.2
        elif volume < 100000:  # Low volume
            score -= 0.4
        
        # Urgency (icebergs take time)
        if urgency == 'high':
            score -= 0.3
        elif urgency == 'low':
            score += 0.2
        
        # Spread (tighter spreads better for iceberg)
        if spread_bps < 10:
            score += 0.2
        elif spread_bps > 25:
            score -= 0.2
        
        return np.clip(score, 0, 1)
    
    def _apply_historical_performance(
        self,
        strategy_scores: Dict[str, float],
        symbol: str,
        market_conditions: Dict[str, Any]
    ) -> Dict[str, float]:
        """Apply historical performance data to strategy scoring"""
        
        adjusted_scores = strategy_scores.copy()
        
        for strategy, base_score in strategy_scores.items():
            if strategy in self.strategy_performance_history:
                recent_performance = self.strategy_performance_history[strategy][-20:]
                
                if len(recent_performance) >= 5:
                    # Calculate average performance
                    avg_slippage = np.mean([p.get('slippage_bps', 10) for p in recent_performance])
                    avg_fill_rate = np.mean([p.get('fill_rate', 0.9) for p in recent_performance])
                    
                    # Performance adjustment
                    performance_score = (
                        (1 - min(avg_slippage / 30, 1)) * 0.5 +  # Lower slippage = better
                        avg_fill_rate * 0.5  # Higher fill rate = better
                    )
                    
                    # Blend with base score
                    weight = self.config.performance_feedback_weight
                    adjusted_scores[strategy] = (
                        (1 - weight) * base_score + weight * performance_score
                    )
        
        return adjusted_scores
    
    def update_strategy_performance(
        self,
        strategy: str,
        performance_metrics: Dict[str, Any]
    ) -> None:
        """Update historical performance data for strategy"""
        
        if strategy not in self.strategy_performance_history:
            self.strategy_performance_history[strategy] = []
        
        performance_record = {
            'timestamp': datetime.now(),
            'slippage_bps': performance_metrics.get('slippage_bps', 0),
            'fill_rate': performance_metrics.get('fill_rate', 0),
            'execution_time': performance_metrics.get('execution_time', 0),
            'market_conditions': performance_metrics.get('market_conditions', {})
        }
        
        self.strategy_performance_history[strategy].append(performance_record)
        
        # Keep only recent history
        if len(self.strategy_performance_history[strategy]) > self.config.learning_window:
            self.strategy_performance_history[strategy] = \
                self.strategy_performance_history[strategy][-self.config.learning_window:]


class VenueSelector:
    """Select optimal venues for order execution"""
    
    def __init__(self, config: RoutingConfig):
        self.config = config
        self.venue_performance_cache: Dict[str, Dict[str, Any]] = {}
        
    def select_venues(
        self,
        available_venues: List[VenueInfo],
        order_characteristics: Dict[str, Any],
        execution_strategy: str
    ) -> Dict[str, float]:
        """
        Select venues and allocate order quantity
        
        Args:
            available_venues: List of available venues
            order_characteristics: Order characteristics
            execution_strategy: Selected execution strategy
            
        Returns:
            Dictionary of venue allocations {venue_name: allocation_ratio}
        """
        
        if not available_venues:
            return {}
        
        # Filter available venues
        active_venues = [v for v in available_venues if v.is_available]
        
        if not active_venues:
            logger.warning("No venues currently available")
            return {}
        
        order_size = order_characteristics.get('size', 0)
        urgency = order_characteristics.get('urgency', 'normal')
        
        # Determine routing approach
        if not self.config.enable_multi_venue_routing or len(active_venues) == 1:
            # Single venue routing
            best_venue = self._select_best_single_venue(active_venues, order_characteristics)
            return {best_venue.name: 1.0}
        
        # Multi-venue routing decision
        if self._should_use_multi_venue(active_venues, order_characteristics, execution_strategy):
            return self._allocate_multi_venue(active_venues, order_characteristics)
        else:
            # Single venue is optimal
            best_venue = self._select_best_single_venue(active_venues, order_characteristics)
            return {best_venue.name: 1.0}
    
    def _select_best_single_venue(
        self,
        venues: List[VenueInfo],
        order_characteristics: Dict[str, Any]
    ) -> VenueInfo:
        """Select single best venue for execution"""
        
        order_size = order_characteristics.get('size', 0)
        urgency = order_characteristics.get('urgency', 'normal')
        
        venue_scores = {}
        
        for venue in venues:
            score = 0.0
            
            # Capacity consideration
            capacity_ratio = min(order_size / venue.capacity, 1.0) if venue.capacity > 0 else 1.0
            capacity_score = 1.0 - capacity_ratio  # Lower ratio = higher score
            score += capacity_score * 0.3
            
            # Cost consideration
            cost_score = 1.0 / (1.0 + venue.avg_slippage_bps / 10)  # Lower slippage = higher score
            score += cost_score * 0.3
            
            # Success rate
            score += venue.success_rate * 0.2
            
            # Latency (important for urgent orders)
            latency_score = 1.0 / (1.0 + venue.latency_ms / 100)
            if urgency == 'high':
                score += latency_score * 0.2
            else:
                score += latency_score * 0.1
            
            venue_scores[venue.name] = score
        
        # Select venue with highest score
        best_venue_name = max(venue_scores, key=venue_scores.get)
        return next(v for v in venues if v.name == best_venue_name)
    
    def _should_use_multi_venue(
        self,
        venues: List[VenueInfo],
        order_characteristics: Dict[str, Any],
        execution_strategy: str
    ) -> bool:
        """Determine if multi-venue routing is beneficial"""
        
        order_size = order_characteristics.get('size', 0)
        
        # Don't fragment small orders
        if order_size < 1000:
            return False
        
        # Check if order exceeds fragmentation threshold for any venue
        for venue in venues:
            if venue.capacity > 0 and order_size > venue.capacity * self.config.fragmentation_threshold:
                return True
        
        # Check if multi-venue can reduce costs
        if len(venues) >= 2:
            # Sort venues by performance
            sorted_venues = sorted(venues, key=lambda v: v.avg_slippage_bps)
            
            # If top 2 venues have similar performance, consider multi-venue
            if len(sorted_venues) >= 2:
                cost_diff = sorted_venues[1].avg_slippage_bps - sorted_venues[0].avg_slippage_bps
                if cost_diff < 5.0:  # Less than 5 bps difference
                    return True
        
        return False
    
    def _allocate_multi_venue(
        self,
        venues: List[VenueInfo],
        order_characteristics: Dict[str, Any]
    ) -> Dict[str, float]:
        """Allocate order across multiple venues"""
        
        order_size = order_characteristics.get('size', 0)
        
        # Sort venues by composite score
        venue_scores = []
        
        for venue in venues:
            # Composite score based on cost, capacity, and reliability
            cost_score = 1.0 / (1.0 + venue.avg_slippage_bps / 10)
            capacity_score = min(venue.capacity / order_size, 1.0) if order_size > 0 else 1.0
            reliability_score = venue.success_rate
            
            composite_score = (cost_score * 0.4 + capacity_score * 0.3 + reliability_score * 0.3)
            
            venue_scores.append((venue, composite_score))
        
        # Sort by score (descending)
        venue_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Allocate to top venues
        allocations = {}
        remaining_allocation = 1.0
        max_venues = min(len(venue_scores), self.config.max_venues_per_order)
        
        for i, (venue, score) in enumerate(venue_scores[:max_venues]):
            if i == max_venues - 1:  # Last venue gets remaining allocation
                allocation = remaining_allocation
            else:
                # Allocate based on relative score and capacity
                base_allocation = score / sum(s for _, s in venue_scores[:max_venues])
                
                # Adjust for capacity constraints
                max_capacity_allocation = min(venue.capacity / order_size, 1.0) if order_size > 0 else 1.0
                allocation = min(base_allocation, max_capacity_allocation)
                
                # Ensure minimum allocation
                allocation = max(allocation, self.config.min_venue_allocation)
                allocation = min(allocation, remaining_allocation)
            
            if allocation > 0:
                allocations[venue.name] = allocation
                remaining_allocation -= allocation
                
                if remaining_allocation <= 0:
                    break
        
        # Normalize allocations to sum to 1.0
        total_allocation = sum(allocations.values())
        if total_allocation > 0:
            allocations = {k: v / total_allocation for k, v in allocations.items()}
        
        return allocations


class SmartOrderRouter:
    """Main smart order routing system"""
    
    def __init__(self, config: RoutingConfig):
        self.config = config
        self.strategy_selector = StrategySelector(config)
        self.venue_selector = VenueSelector(config)
        
        # Initialize component systems
        self.slippage_predictor = None
        self.market_impact_minimizer = None
        self.orderbook_analyzer = None
        self.execution_tracker = None
        
        if config.slippage_prediction_enabled:
            slippage_config = SlippagePredictorConfig()
            self.slippage_predictor = SlippagePredictor(slippage_config)
        
        if config.market_impact_optimization_enabled:
            impact_config = MarketImpactConfig()
            self.market_impact_minimizer = MarketImpactMinimizer(impact_config)
        
        if config.orderbook_analysis_enabled:
            orderbook_config = OrderBookConfig()
            self.orderbook_analyzer = OrderBookAnalyzer(orderbook_config)
        
        if config.execution_tracking_enabled:
            tracker_config = TrackerConfig()
            self.execution_tracker = ExecutionTracker(tracker_config)
        
        # Execution state
        self.active_executions: Dict[str, ExecutionPlan] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        logger.info("Smart Order Router initialized with all components")
    
    async def create_execution_plan(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        urgency: str = "normal",
        available_venues: Optional[List[VenueInfo]] = None,
        market_data: Optional[pd.DataFrame] = None,
        orderbook_snapshot: Optional[OrderBookSnapshot] = None
    ) -> ExecutionPlan:
        """
        Create comprehensive execution plan for order
        
        Args:
            order_id: Unique order identifier
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            urgency: 'low', 'normal', 'high', 'immediate'
            available_venues: List of available venues
            market_data: Historical market data
            orderbook_snapshot: Current order book snapshot
            
        Returns:
            Complete execution plan
        """
        
        logger.info(f"Creating execution plan for order {order_id}: {quantity} {symbol} {side}")
        
        # Prepare order characteristics
        order_characteristics = {
            'symbol': symbol,
            'side': side,
            'size': quantity,
            'urgency': urgency,
            'order_id': order_id
        }
        
        # Analyze market conditions
        market_conditions = await self._analyze_market_conditions(
            symbol, market_data, orderbook_snapshot
        )
        
        # Get venue information
        if available_venues is None:
            available_venues = self._get_default_venues()
        
        venue_conditions = {
            'available_venues': len(available_venues),
            'total_capacity': sum(v.capacity for v in available_venues),
            'avg_latency': np.mean([v.latency_ms for v in available_venues]) if available_venues else 0
        }
        
        # Select execution strategy
        selected_strategy, strategy_confidence = self.strategy_selector.select_strategy(
            order_characteristics, market_conditions, venue_conditions
        )
        
        # Select venues and allocate
        venue_allocations = self.venue_selector.select_venues(
            available_venues, order_characteristics, selected_strategy
        )
        
        # Determine routing decision
        routing_decision = self._determine_routing_decision(
            venue_allocations, order_characteristics, market_conditions
        )
        
        # Generate execution schedule
        execution_schedule = await self._generate_execution_schedule(
            selected_strategy, venue_allocations, order_characteristics, market_conditions
        )
        
        # Predict performance
        expected_performance = await self._predict_execution_performance(
            execution_schedule, market_conditions, orderbook_snapshot
        )
        
        # Assess risks
        risk_assessment = self._assess_execution_risks(
            execution_schedule, market_conditions, expected_performance
        )
        
        # Create contingency plans
        contingency_plans = self._create_contingency_plans(
            order_characteristics, market_conditions, available_venues
        )
        
        # Create execution plan
        execution_plan = ExecutionPlan(
            order_id=order_id,
            routing_decision=routing_decision,
            selected_strategy=selected_strategy,
            venue_allocations=venue_allocations,
            execution_schedule=execution_schedule,
            expected_performance=expected_performance,
            risk_assessment=risk_assessment,
            contingency_plans=contingency_plans
        )
        
        # Store active execution
        self.active_executions[order_id] = execution_plan
        
        logger.info(f"Execution plan created for {order_id}: {selected_strategy} strategy, "
                   f"{len(venue_allocations)} venues, expected cost {execution_plan.total_expected_cost_bps:.1f} bps")
        
        return execution_plan
    
    async def execute_order(
        self,
        execution_plan: ExecutionPlan,
        market_data_feed: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Execute order according to execution plan
        
        Args:
            execution_plan: Complete execution plan
            market_data_feed: Real-time market data feed
            
        Returns:
            Execution results and performance metrics
        """
        
        order_id = execution_plan.order_id
        logger.info(f"Starting execution for order {order_id}")
        
        # Initialize execution tracking
        if self.execution_tracker:
            # Extract order details from execution plan
            first_schedule_item = execution_plan.execution_schedule[0] if execution_plan.execution_schedule else {}
            arrival_price = first_schedule_item.get('arrival_price', 0)
            
            self.execution_tracker.start_order_tracking(
                order_id=order_id,
                symbol=first_schedule_item.get('symbol', ''),
                side=first_schedule_item.get('side', ''),
                quantity=sum(item.get('size', 0) for item in execution_plan.execution_schedule),
                algorithm=execution_plan.selected_strategy,
                arrival_price=arrival_price
            )
        
        execution_results = {
            'order_id': order_id,
            'strategy': execution_plan.selected_strategy,
            'start_time': datetime.now().isoformat(),
            'venue_executions': {},
            'total_filled': 0,
            'total_cost': 0,
            'performance_metrics': {}
        }
        
        try:
            # Execute across venues
            venue_results = await self._execute_across_venues(
                execution_plan, market_data_feed
            )
            
            execution_results['venue_executions'] = venue_results
            
            # Aggregate results
            total_filled = sum(result.get('filled_quantity', 0) for result in venue_results.values())
            total_cost = sum(result.get('total_cost', 0) for result in venue_results.values())
            
            execution_results['total_filled'] = total_filled
            execution_results['total_cost'] = total_cost
            execution_results['fill_rate'] = total_filled / sum(
                item.get('size', 0) for item in execution_plan.execution_schedule
            ) if execution_plan.execution_schedule else 0
            
            # Calculate performance metrics
            performance_metrics = self._calculate_execution_performance(
                execution_plan, venue_results
            )
            execution_results['performance_metrics'] = performance_metrics
            
            # Complete execution tracking
            if self.execution_tracker:
                completion_result = self.execution_tracker.complete_order(order_id)
                execution_results['tracking_metrics'] = completion_result
            
            # Update strategy performance
            self.strategy_selector.update_strategy_performance(
                execution_plan.selected_strategy, performance_metrics
            )
            
            # Store in history
            self.execution_history.append({
                'timestamp': datetime.now(),
                'order_id': order_id,
                'execution_plan': execution_plan,
                'results': execution_results
            })
            
            logger.info(f"Execution completed for {order_id}: "
                       f"Fill rate {execution_results['fill_rate']:.1%}, "
                       f"Total filled {total_filled}")
            
        except Exception as e:
            logger.error(f"Execution failed for order {order_id}: {e}")
            execution_results['error'] = str(e)
            execution_results['status'] = 'failed'
        
        finally:
            # Clean up active execution
            if order_id in self.active_executions:
                del self.active_executions[order_id]
        
        return execution_results
    
    async def _analyze_market_conditions(
        self,
        symbol: str,
        market_data: Optional[pd.DataFrame],
        orderbook_snapshot: Optional[OrderBookSnapshot]
    ) -> Dict[str, Any]:
        """Analyze current market conditions"""
        
        conditions = {
            'symbol': symbol,
            'volatility': 0.02,  # Default values
            'volume': 100000,
            'spread_bps': 10,
            'liquidity_score': 0.5
        }
        
        # Analyze market data
        if market_data is not None and len(market_data) > 20:
            returns = market_data['close'].pct_change().dropna()
            conditions['volatility'] = returns.std() * np.sqrt(252)  # Annualized
            
            if 'volume' in market_data.columns:
                conditions['volume'] = market_data['volume'].tail(20).mean()
        
        # Analyze order book
        if orderbook_snapshot and self.orderbook_analyzer:
            orderbook_analysis = self.orderbook_analyzer.process_snapshot(orderbook_snapshot)
            
            conditions['spread_bps'] = orderbook_snapshot.spread_bps or 10
            conditions['liquidity_score'] = orderbook_analysis['liquidity_analysis']['liquidity_score']
            conditions['order_imbalance'] = orderbook_analysis['liquidity_analysis']['imbalance']
        
        return conditions
    
    async def _generate_execution_schedule(
        self,
        strategy: str,
        venue_allocations: Dict[str, float],
        order_characteristics: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate detailed execution schedule"""
        
        # Create execution config
        execution_config = ExecutionConfig(
            execution_horizon=600 if order_characteristics.get('urgency') == 'normal' else 300,
            max_participation_rate=0.25,
            enable_adaptive_timing=True
        )
        
        # Create strategy instance
        strategy_instance = OrderStrategyFactory.create_strategy(strategy, execution_config)
        
        # Generate mock market data for strategy
        # In real implementation, this would use actual market data
        mock_data = self._create_mock_market_data(market_conditions)
        
        # Generate slices for each venue
        schedule = []
        total_quantity = order_characteristics['size']
        
        for venue_name, allocation_ratio in venue_allocations.items():
            venue_quantity = total_quantity * allocation_ratio
            
            # Generate slices for this venue
            venue_slices = strategy_instance.generate_slices(
                symbol=order_characteristics['symbol'],
                side=OrderSide.BUY if order_characteristics['side'] == 'buy' else OrderSide.SELL,
                quantity=venue_quantity,
                market_data=mock_data
            )
            
            # Convert to schedule format
            for slice_order in venue_slices:
                schedule_item = {
                    'slice_id': slice_order.slice_id,
                    'venue': venue_name,
                    'symbol': order_characteristics['symbol'],
                    'side': order_characteristics['side'],
                    'size': slice_order.quantity,
                    'order_type': slice_order.order_type.value,
                    'scheduled_time': slice_order.timestamp,
                    'price': slice_order.price,
                    'arrival_price': market_conditions.get('current_price', 100)  # Mock price
                }
                schedule.append(schedule_item)
        
        # Sort by scheduled time
        schedule.sort(key=lambda x: x['scheduled_time'])
        
        return schedule
    
    async def _predict_execution_performance(
        self,
        execution_schedule: List[Dict[str, Any]],
        market_conditions: Dict[str, Any],
        orderbook_snapshot: Optional[OrderBookSnapshot]
    ) -> Dict[str, float]:
        """Predict execution performance metrics"""
        
        performance = {
            'slippage_bps': 5.0,  # Default predictions
            'market_impact_bps': 3.0,
            'timing_cost_bps': 2.0,
            'fill_probability': 0.95,
            'expected_duration': 300
        }
        
        # Use slippage predictor if available
        if self.slippage_predictor and self.slippage_predictor.is_trained:
            try:
                # Prepare prediction data
                prediction_data = {
                    'market_data': self._create_mock_market_data(market_conditions),
                    'order_data': pd.DataFrame([{
                        'order_size': sum(item['size'] for item in execution_schedule),
                        'order_side': execution_schedule[0]['side'] if execution_schedule else 'buy',
                        'urgency': 'normal'
                    }])
                }
                
                slippage_prediction = self.slippage_predictor.predict(prediction_data)
                if 'predictions' in slippage_prediction:
                    performance['slippage_bps'] = float(slippage_prediction['predictions'][0]) * 10000
                
            except Exception as e:
                logger.warning(f"Slippage prediction failed: {e}")
        
        # Use market impact minimizer if available
        if self.market_impact_minimizer:
            try:
                total_size = sum(item['size'] for item in execution_schedule)
                impact_estimate = self.market_impact_minimizer.estimate_total_impact(
                    total_size, market_conditions, "optimal"
                )
                
                performance['market_impact_bps'] = impact_estimate['impact_bps']
                
            except Exception as e:
                logger.warning(f"Market impact prediction failed: {e}")
        
        return performance
    
    def _assess_execution_risks(
        self,
        execution_schedule: List[Dict[str, Any]],
        market_conditions: Dict[str, Any],
        expected_performance: Dict[str, float]
    ) -> Dict[str, Any]:
        """Assess execution risks"""
        
        risks = {
            'overall_risk_level': 'medium',
            'key_risks': [],
            'risk_factors': {}
        }
        
        # Market impact risk
        if expected_performance['market_impact_bps'] > 15:
            risks['key_risks'].append('high_market_impact')
            risks['risk_factors']['market_impact'] = expected_performance['market_impact_bps']
        
        # Timing risk
        total_duration = sum(300 for _ in execution_schedule)  # Simplified
        if total_duration > self.config.max_execution_duration:
            risks['key_risks'].append('execution_timeout_risk')
            risks['risk_factors']['duration_risk'] = total_duration
        
        # Volatility risk
        if market_conditions.get('volatility', 0) > 0.30:
            risks['key_risks'].append('high_volatility')
            risks['risk_factors']['volatility'] = market_conditions['volatility']
        
        # Liquidity risk
        if market_conditions.get('liquidity_score', 1) < 0.3:
            risks['key_risks'].append('low_liquidity')
            risks['risk_factors']['liquidity_score'] = market_conditions['liquidity_score']
        
        # Overall risk assessment
        if len(risks['key_risks']) >= 3:
            risks['overall_risk_level'] = 'high'
        elif len(risks['key_risks']) <= 1:
            risks['overall_risk_level'] = 'low'
        
        return risks
    
    def _create_contingency_plans(
        self,
        order_characteristics: Dict[str, Any],
        market_conditions: Dict[str, Any],
        available_venues: List[VenueInfo]
    ) -> List[Dict[str, Any]]:
        """Create contingency plans for execution"""
        
        contingencies = []
        
        # High slippage contingency
        contingencies.append({
            'trigger': 'high_slippage',
            'threshold': 25.0,  # 25 bps
            'action': 'switch_to_iceberg',
            'description': 'Switch to iceberg strategy if slippage exceeds threshold'
        })
        
        # Low fill rate contingency
        contingencies.append({
            'trigger': 'low_fill_rate',
            'threshold': 0.8,  # 80%
            'action': 'increase_aggression',
            'description': 'Increase execution aggression if fill rate is low'
        })
        
        # Venue failure contingency
        if len(available_venues) > 1:
            contingencies.append({
                'trigger': 'venue_failure',
                'threshold': 'venue_unavailable',
                'action': 'reroute_to_backup_venue',
                'description': 'Reroute to backup venue if primary venue fails'
            })
        
        # Market volatility spike contingency
        contingencies.append({
            'trigger': 'volatility_spike',
            'threshold': market_conditions.get('volatility', 0.02) * 2,
            'action': 'pause_and_reassess',
            'description': 'Pause execution and reassess if volatility spikes'
        })
        
        return contingencies
    
    async def _execute_across_venues(
        self,
        execution_plan: ExecutionPlan,
        market_data_feed: Optional[Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Execute order across multiple venues"""
        
        venue_results = {}
        
        # Group execution schedule by venue
        venue_schedules = {}
        for item in execution_plan.execution_schedule:
            venue = item['venue']
            if venue not in venue_schedules:
                venue_schedules[venue] = []
            venue_schedules[venue].append(item)
        
        # Execute on each venue (simulated)
        for venue_name, schedule_items in venue_schedules.items():
            venue_result = await self._execute_on_venue(venue_name, schedule_items)
            venue_results[venue_name] = venue_result
            
            # Record trades if execution tracker is available
            if self.execution_tracker:
                for item in schedule_items:
                    # Simulate trade recording
                    self.execution_tracker.record_trade(
                        trade_id=f"trade_{item['slice_id']}",
                        order_id=execution_plan.order_id,
                        symbol=item['symbol'],
                        side=item['side'],
                        quantity=item['size'] * 0.95,  # Simulate 95% fill
                        price=item.get('price', 100),
                        venue=venue_name,
                        execution_duration=60,  # Simulate 1 minute execution
                        arrival_price=item.get('arrival_price', 100)
                    )
        
        return venue_results
    
    async def _execute_on_venue(
        self,
        venue_name: str,
        schedule_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute order items on specific venue (simulated)"""
        
        # Simulate venue execution
        total_quantity = sum(item['size'] for item in schedule_items)
        filled_quantity = total_quantity * np.random.uniform(0.90, 1.0)  # 90-100% fill rate
        
        avg_price = np.mean([item.get('price', 100) for item in schedule_items])
        slippage_bps = np.random.uniform(2, 10)  # 2-10 bps slippage
        
        execution_time = len(schedule_items) * 30  # 30 seconds per slice
        
        return {
            'venue': venue_name,
            'total_quantity': total_quantity,
            'filled_quantity': filled_quantity,
            'fill_rate': filled_quantity / total_quantity,
            'average_price': avg_price,
            'slippage_bps': slippage_bps,
            'execution_time': execution_time,
            'trades_count': len(schedule_items),
            'total_cost': filled_quantity * avg_price
        }
    
    def _calculate_execution_performance(
        self,
        execution_plan: ExecutionPlan,
        venue_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate overall execution performance"""
        
        # Aggregate venue results
        total_quantity = sum(result['total_quantity'] for result in venue_results.values())
        total_filled = sum(result['filled_quantity'] for result in venue_results.values())
        
        if total_quantity == 0:
            return {}
        
        # Weighted averages
        avg_slippage = sum(
            result['slippage_bps'] * (result['filled_quantity'] / total_filled)
            for result in venue_results.values()
        ) if total_filled > 0 else 0
        
        avg_execution_time = np.mean([result['execution_time'] for result in venue_results.values()])
        
        performance = {
            'overall_fill_rate': total_filled / total_quantity,
            'weighted_avg_slippage_bps': avg_slippage,
            'total_execution_time': avg_execution_time,
            'venues_used': len(venue_results),
            'strategy_used': execution_plan.selected_strategy,
            'vs_expected_slippage': avg_slippage - execution_plan.expected_performance.get('slippage_bps', 0),
            'execution_quality': 'excellent' if avg_slippage < 5 else 'good' if avg_slippage < 15 else 'fair'
        }
        
        return performance
    
    def _determine_routing_decision(
        self,
        venue_allocations: Dict[str, float],
        order_characteristics: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> RoutingDecision:
        """Determine the type of routing decision made"""
        
        num_venues = len(venue_allocations)
        order_size = order_characteristics.get('size', 0)
        
        if num_venues == 1:
            return RoutingDecision.SINGLE_VENUE
        elif num_venues > 1:
            max_allocation = max(venue_allocations.values()) if venue_allocations else 0
            if max_allocation < 0.7:  # No single venue dominates
                return RoutingDecision.FRAGMENTED
            else:
                return RoutingDecision.MULTI_VENUE
        
        # Default case
        return RoutingDecision.SINGLE_VENUE
    
    def _get_default_venues(self) -> List[VenueInfo]:
        """Get default venue list (for demo purposes)"""
        
        return [
            VenueInfo(
                name="NASDAQ",
                liquidity=1000000,
                latency_ms=5,
                cost_per_share=0.001,
                success_rate=0.98,
                avg_slippage_bps=8.5,
                capacity=500000
            ),
            VenueInfo(
                name="NYSE",
                liquidity=1200000,
                latency_ms=8,
                cost_per_share=0.0012,
                success_rate=0.97,
                avg_slippage_bps=9.2,
                capacity=600000
            ),
            VenueInfo(
                name="BATS",
                liquidity=800000,
                latency_ms=3,
                cost_per_share=0.0008,
                success_rate=0.96,
                avg_slippage_bps=7.8,
                capacity=400000
            )
        ]
    
    def _create_mock_market_data(self, market_conditions: Dict[str, Any]) -> pd.DataFrame:
        """Create mock market data for testing"""
        
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        
        # Generate synthetic OHLCV data
        base_price = 100
        volatility = market_conditions.get('volatility', 0.02) / np.sqrt(252 * 24)  # Hourly vol
        
        returns = np.random.normal(0, volatility, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        volumes = np.random.normal(
            market_conditions.get('volume', 100000),
            market_conditions.get('volume', 100000) * 0.3,
            len(dates)
        )
        volumes = np.maximum(volumes, 1000)  # Minimum volume
        
        data = pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.001, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.001, len(dates)))),
            'close': prices,
            'volume': volumes
        }, index=dates)
        
        return data
    
    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get comprehensive routing analytics"""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'active_executions': len(self.active_executions),
            'total_executions': len(self.execution_history),
            'strategy_performance': self.strategy_selector.strategy_performance_history,
            'venue_performance': getattr(self.venue_selector, 'venue_performance_cache', {}),
            'recent_execution_summary': self._get_recent_execution_summary(),
            'system_status': {
                'slippage_predictor': 'enabled' if self.slippage_predictor else 'disabled',
                'market_impact_optimizer': 'enabled' if self.market_impact_minimizer else 'disabled',
                'orderbook_analyzer': 'enabled' if self.orderbook_analyzer else 'disabled',
                'execution_tracker': 'enabled' if self.execution_tracker else 'disabled'
            }
        }
    
    def _get_recent_execution_summary(self) -> Dict[str, Any]:
        """Get summary of recent executions"""
        
        if not self.execution_history:
            return {'status': 'no_recent_executions'}
        
        recent_executions = self.execution_history[-20:]  # Last 20 executions
        
        fill_rates = []
        slippages = []
        strategies_used = []
        
        for execution in recent_executions:
            results = execution['results']
            fill_rates.append(results.get('fill_rate', 0))
            
            if 'performance_metrics' in results:
                slippages.append(results['performance_metrics'].get('weighted_avg_slippage_bps', 0))
            
            strategies_used.append(execution['execution_plan'].selected_strategy)
        
        return {
            'executions_analyzed': len(recent_executions),
            'avg_fill_rate': np.mean(fill_rates) if fill_rates else 0,
            'avg_slippage_bps': np.mean(slippages) if slippages else 0,
            'strategy_distribution': {
                strategy: strategies_used.count(strategy)
                for strategy in set(strategies_used)
            },
            'success_rate': len([fr for fr in fill_rates if fr > 0.95]) / len(fill_rates) if fill_rates else 0
        }