"""
Advanced Order Book Analysis

Real-time order book analysis for optimal execution:
- Level-2 order book depth analysis and liquidity assessment
- Market microstructure pattern recognition and prediction
- Queue position estimation and fill probability modeling
- Bid-ask spread dynamics and timing optimization
- Smart order placement with queue jumping detection
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import deque
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class OrderBookConfig:
    """Configuration for order book analysis"""
    
    # Analysis parameters
    max_levels: int = 10                    # Maximum order book levels to analyze
    min_levels: int = 3                     # Minimum levels required for analysis
    update_frequency_ms: int = 100          # Order book update frequency
    
    # Liquidity analysis
    depth_analysis_levels: int = 5          # Levels for depth analysis
    imbalance_threshold: float = 0.7        # Threshold for order flow imbalance
    liquidity_decay_factor: float = 0.9     # Decay factor for distant levels
    
    # Microstructure parameters
    tick_size: float = 0.01                 # Minimum price increment
    queue_analysis_enabled: bool = True     # Enable queue position analysis
    spread_prediction_window: int = 50      # Window for spread prediction
    
    # Pattern recognition
    pattern_detection_enabled: bool = True   # Enable pattern detection
    pattern_lookback: int = 100             # Lookback for pattern analysis  
    momentum_threshold: float = 0.6         # Threshold for momentum patterns
    
    # Risk parameters
    max_order_size_ratio: float = 0.3       # Max order size vs level size
    adverse_selection_penalty: float = 0.001 # Penalty for adverse selection
    
    # Performance tracking
    track_execution_quality: bool = True     # Track execution quality metrics
    quality_measurement_window: int = 20     # Window for quality measurement
    
    def __post_init__(self):
        """Validate configuration"""
        if self.max_levels < self.min_levels:
            raise ValueError("max_levels must be >= min_levels")
        if not (0 < self.imbalance_threshold < 1):
            raise ValueError("imbalance_threshold must be between 0 and 1")


@dataclass
class OrderBookLevel:
    """Single level in the order book"""
    price: float
    size: float
    orders: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def avg_order_size(self) -> float:
        """Average order size at this level"""
        return self.size / max(1, self.orders)


@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot"""
    timestamp: datetime
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    sequence_number: Optional[int] = None
    
    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        """Best bid level"""
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        """Best ask level"""
        return self.asks[0] if self.asks else None
    
    @property
    def mid_price(self) -> Optional[float]:
        """Mid price"""
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2
        return None
    
    @property
    def spread(self) -> Optional[float]:
        """Bid-ask spread"""
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return None
    
    @property
    def spread_bps(self) -> Optional[float]:
        """Spread in basis points"""
        if self.spread and self.mid_price:
            return (self.spread / self.mid_price) * 10000
        return None


class LiquidityAnalyzer:
    """Analyze order book liquidity and depth"""
    
    def __init__(self, config: OrderBookConfig):
        self.config = config
        
    def analyze_depth(self, snapshot: OrderBookSnapshot) -> Dict[str, Any]:
        """
        Analyze order book depth and liquidity
        
        Args:
            snapshot: Order book snapshot
            
        Returns:
            Depth analysis results
        """
        
        analysis = {
            'timestamp': snapshot.timestamp.isoformat(),
            'symbol': snapshot.symbol,
            'bid_depth': self._calculate_side_depth(snapshot.bids),
            'ask_depth': self._calculate_side_depth(snapshot.asks),
            'imbalance': self._calculate_imbalance(snapshot),
            'liquidity_score': 0.0,
            'effective_spread': self._calculate_effective_spread(snapshot)
        }
        
        # Calculate overall liquidity score
        bid_depth = analysis['bid_depth']
        ask_depth = analysis['ask_depth']
        
        depth_score = min(bid_depth['total_size'], ask_depth['total_size']) / 10000  # Normalize
        levels_score = min(len(snapshot.bids), len(snapshot.asks)) / self.config.max_levels
        spread_score = max(0, 1 - (analysis['effective_spread'] or 0.01))
        
        analysis['liquidity_score'] = (depth_score + levels_score + spread_score) / 3
        
        return analysis
    
    def _calculate_side_depth(self, levels: List[OrderBookLevel]) -> Dict[str, Any]:
        """Calculate depth metrics for one side of the book"""
        
        if not levels:
            return {'total_size': 0, 'weighted_price': 0, 'level_count': 0, 'avg_size': 0}
        
        # Analyze up to configured number of levels
        analyzed_levels = levels[:self.config.depth_analysis_levels]
        
        total_size = 0
        weighted_price_sum = 0
        
        for i, level in enumerate(analyzed_levels):
            # Apply decay factor for distant levels
            decay_factor = self.config.liquidity_decay_factor ** i
            effective_size = level.size * decay_factor
            
            total_size += effective_size
            weighted_price_sum += level.price * effective_size
        
        return {
            'total_size': total_size,
            'weighted_price': weighted_price_sum / total_size if total_size > 0 else 0,
            'level_count': len(analyzed_levels),
            'avg_size': total_size / len(analyzed_levels) if analyzed_levels else 0,
            'top_level_size': levels[0].size,
            'size_distribution': [level.size for level in analyzed_levels[:5]]
        }
    
    def _calculate_imbalance(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """Calculate order flow imbalance"""
        
        if not snapshot.bids or not snapshot.asks:
            return {'ratio': 0.5, 'strength': 0.0, 'direction': 'neutral'}
        
        # Calculate size-weighted imbalance
        bid_size = sum(level.size for level in snapshot.bids[:3])
        ask_size = sum(level.size for level in snapshot.asks[:3])
        
        total_size = bid_size + ask_size
        if total_size == 0:
            return {'ratio': 0.5, 'strength': 0.0, 'direction': 'neutral'}
        
        bid_ratio = bid_size / total_size
        
        # Determine direction and strength
        if bid_ratio > self.config.imbalance_threshold:
            direction = 'bullish'
            strength = (bid_ratio - 0.5) * 2  # Scale to 0-1
        elif bid_ratio < (1 - self.config.imbalance_threshold):
            direction = 'bearish'
            strength = (0.5 - bid_ratio) * 2
        else:
            direction = 'neutral'
            strength = 0.0
        
        return {
            'ratio': bid_ratio,
            'strength': strength,
            'direction': direction,
            'bid_size': bid_size,
            'ask_size': ask_size
        }
    
    def _calculate_effective_spread(self, snapshot: OrderBookSnapshot) -> Optional[float]:
        """Calculate effective spread considering order sizes"""
        
        if not snapshot.best_bid or not snapshot.best_ask:
            return None
        
        # Weight spread by available liquidity
        bid_size = snapshot.best_bid.size
        ask_size = snapshot.best_ask.size
        total_size = bid_size + ask_size
        
        if total_size == 0:
            return snapshot.spread_bps
        
        # Effective spread accounts for size imbalance
        size_penalty = abs(bid_size - ask_size) / total_size * 0.001  # Small penalty
        
        return (snapshot.spread_bps or 0) + size_penalty * 10000
    
    def estimate_market_impact(
        self,
        snapshot: OrderBookSnapshot,
        order_size: float,
        side: str
    ) -> Dict[str, Any]:
        """
        Estimate market impact of an order given current book state
        
        Args:
            snapshot: Current order book snapshot
            order_size: Size of order to analyze
            side: 'buy' or 'sell'
            
        Returns:
            Market impact estimation
        """
        
        levels = snapshot.asks if side == 'buy' else snapshot.bids
        
        if not levels:
            return {'impact': float('inf'), 'feasible': False}
        
        cumulative_size = 0
        weighted_price_sum = 0
        levels_consumed = 0
        
        reference_price = snapshot.mid_price or levels[0].price
        
        for level in levels:
            available_size = min(level.size, order_size - cumulative_size)
            
            cumulative_size += available_size
            weighted_price_sum += level.price * available_size
            levels_consumed += 1
            
            if cumulative_size >= order_size:
                break
        
        if cumulative_size < order_size:
            # Order cannot be fully filled
            return {
                'impact': float('inf'),
                'feasible': False,
                'available_liquidity': cumulative_size,
                'shortage': order_size - cumulative_size
            }
        
        # Calculate weighted average execution price
        avg_execution_price = weighted_price_sum / cumulative_size
        
        # Impact as percentage of reference price
        impact_pct = abs(avg_execution_price - reference_price) / reference_price
        
        return {
            'impact': impact_pct,
            'impact_bps': impact_pct * 10000,
            'feasible': True,
            'avg_execution_price': avg_execution_price,
            'levels_consumed': levels_consumed,
            'reference_price': reference_price,
            'liquidity_consumed': cumulative_size / sum(l.size for l in levels[:5])
        }


class MicrostructureAnalyzer:
    """Analyze market microstructure patterns"""
    
    def __init__(self, config: OrderBookConfig):
        self.config = config
        self.snapshot_history: deque = deque(maxlen=config.pattern_lookback)
        self.pattern_cache: Dict[str, Any] = {}
        
    def add_snapshot(self, snapshot: OrderBookSnapshot) -> None:
        """Add new snapshot to history"""
        self.snapshot_history.append(snapshot)
        
    def detect_patterns(self) -> Dict[str, Any]:
        """
        Detect microstructure patterns in recent snapshots
        
        Returns:
            Detected patterns and their characteristics
        """
        
        if len(self.snapshot_history) < 10:
            return {'patterns': [], 'confidence': 0.0}
        
        patterns = []
        
        # Pattern 1: Spread tightening/widening
        spread_pattern = self._detect_spread_pattern()
        if spread_pattern:
            patterns.append(spread_pattern)
        
        # Pattern 2: Liquidity buildup/depletion
        liquidity_pattern = self._detect_liquidity_pattern()
        if liquidity_pattern:
            patterns.append(liquidity_pattern)
        
        # Pattern 3: Order flow momentum
        momentum_pattern = self._detect_momentum_pattern()
        if momentum_pattern:
            patterns.append(momentum_pattern)
        
        # Pattern 4: Level stepping (walk up/down)
        stepping_pattern = self._detect_stepping_pattern()
        if stepping_pattern:
            patterns.append(stepping_pattern)
        
        # Calculate overall confidence
        confidence = min(len(patterns) / 3, 1.0) if patterns else 0.0
        
        return {
            'patterns': patterns,
            'pattern_count': len(patterns),
            'confidence': confidence,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _detect_spread_pattern(self) -> Optional[Dict[str, Any]]:
        """Detect spread tightening/widening patterns"""
        
        recent_snapshots = list(self.snapshot_history)[-20:]
        spreads = [s.spread_bps for s in recent_snapshots if s.spread_bps is not None]
        
        if len(spreads) < 10:
            return None
        
        # Calculate trend
        x = np.arange(len(spreads))
        trend = np.polyfit(x, spreads, 1)[0]  # Linear trend coefficient
        
        # Statistical significance
        spread_std = np.std(spreads)
        trend_significance = abs(trend) / (spread_std + 1e-6)
        
        if trend_significance > 0.5:  # Significant trend
            pattern_type = 'spread_tightening' if trend < 0 else 'spread_widening'
            
            return {
                'type': pattern_type,
                'trend_strength': trend_significance,
                'current_spread': spreads[-1],
                'avg_spread': np.mean(spreads),
                'prediction': 'continued_trend' if trend_significance > 1.0 else 'reversal_likely',
                'confidence': min(trend_significance, 1.0)
            }
        
        return None
    
    def _detect_liquidity_pattern(self) -> Optional[Dict[str, Any]]:
        """Detect liquidity buildup or depletion patterns"""
        
        recent_snapshots = list(self.snapshot_history)[-15:]
        
        bid_sizes = []
        ask_sizes = []
        
        for snapshot in recent_snapshots:
            bid_size = sum(level.size for level in snapshot.bids[:3])
            ask_size = sum(level.size for level in snapshot.asks[:3])
            bid_sizes.append(bid_size)
            ask_sizes.append(ask_size)
        
        if len(bid_sizes) < 8:
            return None
        
        # Calculate trends
        x = np.arange(len(bid_sizes))
        bid_trend = np.polyfit(x, bid_sizes, 1)[0]
        ask_trend = np.polyfit(x, ask_sizes, 1)[0]
        
        # Determine pattern
        bid_change = (bid_sizes[-1] - bid_sizes[0]) / (bid_sizes[0] + 1)
        ask_change = (ask_sizes[-1] - ask_sizes[0]) / (ask_sizes[0] + 1)
        
        threshold = 0.2  # 20% change threshold
        
        if abs(bid_change) > threshold or abs(ask_change) > threshold:
            if bid_change > threshold and ask_change > threshold:
                pattern_type = 'liquidity_buildup'
            elif bid_change < -threshold and ask_change < -threshold:
                pattern_type = 'liquidity_depletion'
            elif bid_change > threshold:
                pattern_type = 'bid_buildup'
            elif ask_change > threshold:
                pattern_type = 'ask_buildup'
            elif bid_change < -threshold:
                pattern_type = 'bid_depletion'
            else:
                pattern_type = 'ask_depletion'
            
            return {
                'type': pattern_type,
                'bid_change_pct': bid_change,
                'ask_change_pct': ask_change,
                'current_bid_size': bid_sizes[-1],
                'current_ask_size': ask_sizes[-1],
                'confidence': min(max(abs(bid_change), abs(ask_change)), 1.0)
            }
        
        return None
    
    def _detect_momentum_pattern(self) -> Optional[Dict[str, Any]]:
        """Detect order flow momentum patterns"""
        
        recent_snapshots = list(self.snapshot_history)[-10:]
        
        if len(recent_snapshots) < 5:
            return None
        
        # Calculate price momentum
        prices = [s.mid_price for s in recent_snapshots if s.mid_price is not None]
        
        if len(prices) < 5:
            return None
        
        # Price change momentum
        price_changes = np.diff(prices)
        recent_momentum = np.mean(price_changes[-3:])  # Last 3 changes
        overall_momentum = np.mean(price_changes)
        
        # Order flow imbalance momentum
        imbalances = []
        for snapshot in recent_snapshots:
            if snapshot.bids and snapshot.asks:
                bid_size = sum(l.size for l in snapshot.bids[:2])
                ask_size = sum(l.size for l in snapshot.asks[:2])
                imbalance = (bid_size - ask_size) / (bid_size + ask_size + 1e-6)
                imbalances.append(imbalance)
        
        if len(imbalances) >= 5:
            imbalance_momentum = np.mean(imbalances[-3:])
            
            # Determine momentum strength
            momentum_strength = abs(recent_momentum) / (np.std(price_changes) + 1e-6)
            imbalance_strength = abs(imbalance_momentum)
            
            if momentum_strength > self.config.momentum_threshold or imbalance_strength > 0.3:
                direction = 'bullish' if recent_momentum > 0 else 'bearish'
                
                return {
                    'type': 'momentum_pattern',
                    'direction': direction,
                    'price_momentum': recent_momentum,
                    'imbalance_momentum': imbalance_momentum,
                    'strength': momentum_strength,
                    'confidence': min(momentum_strength + imbalance_strength, 1.0)
                }
        
        return None
    
    def _detect_stepping_pattern(self) -> Optional[Dict[str, Any]]:
        """Detect level stepping patterns (systematic level movement)"""
        
        recent_snapshots = list(self.snapshot_history)[-12:]
        
        if len(recent_snapshots) < 8:
            return None
        
        # Track best bid/ask movements
        best_bids = [s.best_bid.price for s in recent_snapshots if s.best_bid]
        best_asks = [s.best_ask.price for s in recent_snapshots if s.best_ask]
        
        if len(best_bids) < 8 or len(best_asks) < 8:
            return None
        
        # Count directional moves
        bid_moves = np.diff(best_bids)
        ask_moves = np.diff(best_asks)
        
        # Look for consistent stepping
        bid_up_moves = np.sum(bid_moves > 0)
        bid_down_moves = np.sum(bid_moves < 0)
        ask_up_moves = np.sum(ask_moves > 0)
        ask_down_moves = np.sum(ask_moves < 0)
        
        total_moves = len(bid_moves)
        
        # Pattern detection thresholds
        stepping_threshold = 0.7  # 70% of moves in same direction
        
        patterns_detected = []
        
        if bid_up_moves / total_moves > stepping_threshold:
            patterns_detected.append('bid_stepping_up')
        elif bid_down_moves / total_moves > stepping_threshold:
            patterns_detected.append('bid_stepping_down')
        
        if ask_up_moves / total_moves > stepping_threshold:
            patterns_detected.append('ask_stepping_up')
        elif ask_down_moves / total_moves > stepping_threshold:
            patterns_detected.append('ask_stepping_down')
        
        if patterns_detected:
            return {
                'type': 'stepping_pattern',
                'patterns': patterns_detected,
                'bid_direction_bias': (bid_up_moves - bid_down_moves) / total_moves,
                'ask_direction_bias': (ask_up_moves - ask_down_moves) / total_moves,
                'consistency': max(bid_up_moves, bid_down_moves, ask_up_moves, ask_down_moves) / total_moves,
                'confidence': max(bid_up_moves, bid_down_moves, ask_up_moves, ask_down_moves) / total_moves
            }
        
        return None


class QueuePositionEstimator:
    """Estimate queue position and fill probabilities"""
    
    def __init__(self, config: OrderBookConfig):
        self.config = config
        self.order_tracking: Dict[str, Dict[str, Any]] = {}
        
    def estimate_queue_position(
        self,
        order_price: float,
        order_side: str,
        snapshot: OrderBookSnapshot,
        order_timestamp: datetime
    ) -> Dict[str, Any]:
        """
        Estimate queue position for a limit order
        
        Args:
            order_price: Price of the limit order
            order_side: 'buy' or 'sell'
            snapshot: Current order book snapshot
            order_timestamp: When order was placed
            
        Returns:
            Queue position estimate and fill probability
        """
        
        levels = snapshot.bids if order_side == 'buy' else snapshot.asks
        
        # Find the level where order would be placed
        target_level = None
        position_in_queue = 0
        
        for level in levels:
            if (order_side == 'buy' and level.price == order_price) or \
               (order_side == 'sell' and level.price == order_price):
                target_level = level
                break
        
        if target_level is None:
            # Order price not in current book
            return {
                'queue_position': 'unknown',
                'estimated_position': 0,
                'fill_probability': 0.0,
                'expected_fill_time': None,
                'level_found': False
            }
        
        # Estimate position based on order timing and level dynamics
        # Simplified model: assume orders are filled in time priority
        
        # Estimate how much was already at this level when order was placed
        time_since_order = (snapshot.timestamp - order_timestamp).total_seconds()
        
        # Simple model: assume some portion of current level was there before our order
        estimated_ahead = target_level.size * 0.6  # 60% of current size was ahead
        
        # Adjust based on time - newer orders have less ahead of them
        if time_since_order < 60:  # Less than 1 minute
            time_factor = 1 - (time_since_order / 120)  # Reduce estimate for recent orders
            estimated_ahead *= time_factor
        
        # Calculate fill probability based on position and market activity
        fill_probability = self._calculate_fill_probability(
            estimated_ahead, target_level.size, order_side, snapshot
        )
        
        # Estimate fill time
        expected_fill_time = self._estimate_fill_time(
            estimated_ahead, target_level, order_side, snapshot
        )
        
        return {
            'queue_position': int(estimated_ahead),
            'estimated_position': estimated_ahead,
            'level_size': target_level.size,
            'fill_probability': fill_probability,
            'expected_fill_time': expected_fill_time,
            'level_found': True,
            'level_price': target_level.price
        }
    
    def _calculate_fill_probability(
        self,
        position_in_queue: float,
        level_size: float,
        order_side: str,
        snapshot: OrderBookSnapshot
    ) -> float:
        """Calculate probability of order being filled"""
        
        if level_size == 0:
            return 0.0
        
        # Base probability from queue position
        queue_ratio = position_in_queue / level_size
        base_prob = max(0, 1 - queue_ratio)
        
        # Adjust for market conditions
        if snapshot.spread_bps:
            # Tighter spreads = higher fill probability
            spread_factor = max(0.5, 1 - (snapshot.spread_bps - 5) / 20)
            base_prob *= spread_factor
        
        # Adjust for order flow imbalance
        if snapshot.bids and snapshot.asks:
            bid_size = sum(l.size for l in snapshot.bids[:2])
            ask_size = sum(l.size for l in snapshot.asks[:2])
            
            if order_side == 'buy':
                # Buy orders more likely to fill if ask side is thinner
                imbalance_factor = bid_size / (bid_size + ask_size)
                base_prob *= (1 + imbalance_factor * 0.2)
            else:
                # Sell orders more likely to fill if bid side is thicker
                imbalance_factor = ask_size / (bid_size + ask_size)
                base_prob *= (1 + imbalance_factor * 0.2)
        
        return min(1.0, base_prob)
    
    def _estimate_fill_time(
        self,
        position_in_queue: float,
        level: OrderBookLevel,
        order_side: str,
        snapshot: OrderBookSnapshot
    ) -> Optional[int]:
        """Estimate time until order fill in seconds"""
        
        if position_in_queue <= 0:
            return 0  # Immediate fill
        
        # Estimate fill rate based on recent activity
        # This is a simplified model - in practice would use historical fill rates
        
        base_fill_rate = level.size / 300  # Assume level turns over every 5 minutes
        
        # Adjust for market conditions
        if snapshot.spread_bps and snapshot.spread_bps < 10:
            # Tight spreads = faster fills
            base_fill_rate *= 1.5
        
        # Adjust for level size
        if level.size > 10000:  # Large level
            base_fill_rate *= 0.7  # Slower turnover
        
        if base_fill_rate > 0:
            estimated_seconds = position_in_queue / base_fill_rate
            return int(min(estimated_seconds, 3600))  # Cap at 1 hour
        
        return None


class OrderBookAnalyzer:
    """Main order book analyzer combining all analysis components"""
    
    def __init__(self, config: OrderBookConfig):
        self.config = config
        self.liquidity_analyzer = LiquidityAnalyzer(config)
        self.microstructure_analyzer = MicrostructureAnalyzer(config)
        self.queue_estimator = QueuePositionEstimator(config)
        
        self.analysis_history: deque = deque(maxlen=1000)
        self.performance_metrics: Dict[str, Any] = {}
        
    def process_snapshot(self, snapshot: OrderBookSnapshot) -> Dict[str, Any]:
        """
        Process new order book snapshot and perform comprehensive analysis
        
        Args:
            snapshot: Order book snapshot to analyze
            
        Returns:
            Comprehensive analysis results
        """
        
        logger.info(f"Processing order book snapshot for {snapshot.symbol}")
        
        # Add to microstructure analyzer
        self.microstructure_analyzer.add_snapshot(snapshot)
        
        # Perform all analyses
        analysis_result = {
            'timestamp': snapshot.timestamp.isoformat(),
            'symbol': snapshot.symbol,
            'snapshot_data': self._extract_snapshot_summary(snapshot),
            'liquidity_analysis': self.liquidity_analyzer.analyze_depth(snapshot),
            'microstructure_patterns': self.microstructure_analyzer.detect_patterns(),
            'trading_signals': self._generate_trading_signals(snapshot),
            'execution_recommendations': self._generate_execution_recommendations(snapshot)
        }
        
        # Store in history
        self.analysis_history.append(analysis_result)
        
        return analysis_result
    
    def analyze_order_placement(
        self,
        order_size: float,
        order_side: str,
        snapshot: OrderBookSnapshot,
        placement_strategy: str = "optimal"
    ) -> Dict[str, Any]:
        """
        Analyze optimal order placement given current market conditions
        
        Args:
            order_size: Size of order to place
            order_side: 'buy' or 'sell'
            snapshot: Current order book snapshot
            placement_strategy: Strategy for order placement
            
        Returns:
            Order placement recommendations
        """
        
        # Market impact analysis
        market_impact = self.liquidity_analyzer.estimate_market_impact(
            snapshot, order_size, order_side
        )
        
        # Generate placement recommendations
        recommendations = []
        
        if market_impact['feasible']:
            # Strategy 1: Market order
            market_rec = {
                'strategy': 'market_order',
                'expected_impact': market_impact['impact_bps'],
                'execution_certainty': 1.0,
                'expected_fill_time': 0,
                'pros': ['Immediate execution', 'No queue risk'],
                'cons': ['Higher cost', 'Market impact']
            }
            recommendations.append(market_rec)
            
            # Strategy 2: Best bid/ask
            if order_side == 'buy' and snapshot.best_ask:
                queue_analysis = self.queue_estimator.estimate_queue_position(
                    snapshot.best_ask.price, order_side, snapshot, datetime.now()
                )
                
                best_ask_rec = {
                    'strategy': 'join_best_ask',
                    'price': snapshot.best_ask.price,
                    'expected_impact': 0,  # No immediate impact
                    'execution_certainty': queue_analysis['fill_probability'],
                    'expected_fill_time': queue_analysis['expected_fill_time'],
                    'queue_position': queue_analysis['estimated_position'],
                    'pros': ['No market impact', 'Better price'],
                    'cons': ['Execution uncertainty', 'Queue risk']
                }
                recommendations.append(best_ask_rec)
            
            elif order_side == 'sell' and snapshot.best_bid:
                queue_analysis = self.queue_estimator.estimate_queue_position(
                    snapshot.best_bid.price, order_side, snapshot, datetime.now()
                )
                
                best_bid_rec = {
                    'strategy': 'join_best_bid',
                    'price': snapshot.best_bid.price,
                    'expected_impact': 0,
                    'execution_certainty': queue_analysis['fill_probability'],
                    'expected_fill_time': queue_analysis['expected_fill_time'],
                    'queue_position': queue_analysis['estimated_position'],
                    'pros': ['No market impact', 'Better price'],
                    'cons': ['Execution uncertainty', 'Queue risk']
                }
                recommendations.append(best_bid_rec)
            
            # Strategy 3: Mid-price order (if spread is wide)
            if snapshot.spread_bps and snapshot.spread_bps > 10:
                mid_price = snapshot.mid_price
                mid_rec = {
                    'strategy': 'mid_price_order',
                    'price': mid_price,
                    'expected_impact': market_impact['impact_bps'] * 0.5,
                    'execution_certainty': 0.3,  # Lower certainty
                    'expected_fill_time': 300,  # Estimated 5 minutes
                    'pros': ['Price improvement', 'Reduced impact'],
                    'cons': ['Low fill probability', 'Longer wait time']
                }
                recommendations.append(mid_rec)
        
        # Select optimal strategy
        optimal_strategy = self._select_optimal_strategy(
            recommendations, placement_strategy, order_size
        )
        
        return {
            'order_size': order_size,
            'order_side': order_side,
            'market_impact_analysis': market_impact,
            'recommendations': recommendations,
            'optimal_strategy': optimal_strategy,
            'market_conditions': {
                'spread_bps': snapshot.spread_bps,
                'liquidity_score': self.liquidity_analyzer.analyze_depth(snapshot)['liquidity_score'],
                'imbalance': self.liquidity_analyzer._calculate_imbalance(snapshot)
            }
        }
    
    def _extract_snapshot_summary(self, snapshot: OrderBookSnapshot) -> Dict[str, Any]:
        """Extract key snapshot data"""
        
        return {
            'best_bid': snapshot.best_bid.price if snapshot.best_bid else None,
            'best_ask': snapshot.best_ask.price if snapshot.best_ask else None,
            'mid_price': snapshot.mid_price,
            'spread': snapshot.spread,
            'spread_bps': snapshot.spread_bps,
            'bid_levels': len(snapshot.bids),
            'ask_levels': len(snapshot.asks),
            'total_bid_size': sum(l.size for l in snapshot.bids[:5]),
            'total_ask_size': sum(l.size for l in snapshot.asks[:5])
        }
    
    def _generate_trading_signals(self, snapshot: OrderBookSnapshot) -> Dict[str, Any]:
        """Generate trading signals based on order book analysis"""
        
        signals = []
        
        # Signal 1: Liquidity imbalance
        imbalance = self.liquidity_analyzer._calculate_imbalance(snapshot)
        
        if imbalance['strength'] > 0.7:
            signals.append({
                'type': 'liquidity_imbalance',
                'direction': imbalance['direction'],
                'strength': imbalance['strength'],
                'confidence': 0.6
            })
        
        # Signal 2: Spread anomaly
        if snapshot.spread_bps:
            # Use historical context if available
            recent_spreads = [
                analysis['snapshot_data']['spread_bps']
                for analysis in list(self.analysis_history)[-20:]
                if analysis['snapshot_data']['spread_bps'] is not None
            ]
            
            if recent_spreads:
                avg_spread = np.mean(recent_spreads)
                
                if snapshot.spread_bps < avg_spread * 0.7:  # Unusually tight
                    signals.append({
                        'type': 'tight_spread',
                        'current_spread': snapshot.spread_bps,
                        'avg_spread': avg_spread,
                        'confidence': 0.7
                    })
                elif snapshot.spread_bps > avg_spread * 1.5:  # Unusually wide
                    signals.append({
                        'type': 'wide_spread',
                        'current_spread': snapshot.spread_bps,
                        'avg_spread': avg_spread,
                        'confidence': 0.7
                    })
        
        # Signal 3: Level size anomaly
        if snapshot.best_bid and snapshot.best_ask:
            size_ratio = snapshot.best_bid.size / snapshot.best_ask.size
            
            if size_ratio > 3:  # Bid much larger
                signals.append({
                    'type': 'bid_size_dominance',
                    'size_ratio': size_ratio,
                    'confidence': 0.5
                })
            elif size_ratio < 1/3:  # Ask much larger
                signals.append({
                    'type': 'ask_size_dominance',
                    'size_ratio': size_ratio,
                    'confidence': 0.5
                })
        
        return {
            'signals': signals,
            'signal_count': len(signals),
            'overall_sentiment': self._aggregate_signal_sentiment(signals)
        }
    
    def _generate_execution_recommendations(self, snapshot: OrderBookSnapshot) -> Dict[str, Any]:
        """Generate execution strategy recommendations"""
        
        recommendations = {
            'market_conditions': 'normal',
            'recommended_strategies': [],
            'risk_factors': [],
            'timing_advice': 'neutral'
        }
        
        # Assess market conditions
        liquidity = self.liquidity_analyzer.analyze_depth(snapshot)
        
        if liquidity['liquidity_score'] < 0.3:
            recommendations['market_conditions'] = 'illiquid'
            recommendations['recommended_strategies'].extend(['iceberg', 'twap'])
            recommendations['risk_factors'].append('low_liquidity')
        elif liquidity['liquidity_score'] > 0.8:
            recommendations['market_conditions'] = 'liquid'
            recommendations['recommended_strategies'].extend(['market', 'vwap'])
        
        # Check for patterns
        patterns = self.microstructure_analyzer.detect_patterns()
        
        for pattern in patterns.get('patterns', []):
            if pattern['type'] == 'momentum_pattern':
                if pattern['direction'] == 'bullish':
                    recommendations['timing_advice'] = 'buy_urgency'
                else:
                    recommendations['timing_advice'] = 'sell_urgency'
            
            elif pattern['type'] == 'spread_tightening':
                recommendations['recommended_strategies'].append('limit_orders')
            
            elif pattern['type'] == 'liquidity_depletion':
                recommendations['risk_factors'].append('liquidity_risk')
                recommendations['recommended_strategies'] = ['iceberg', 'twap']
        
        return recommendations
    
    def _select_optimal_strategy(
        self,
        recommendations: List[Dict[str, Any]],
        placement_strategy: str,
        order_size: float
    ) -> Optional[Dict[str, Any]]:
        """Select optimal placement strategy from recommendations"""
        
        if not recommendations:
            return None
        
        if placement_strategy == "aggressive":
            # Prefer high certainty, fast execution
            return max(recommendations, key=lambda x: x['execution_certainty'])
        
        elif placement_strategy == "passive":
            # Prefer low impact, better prices
            return min(recommendations, key=lambda x: x['expected_impact'])
        
        else:  # optimal
            # Balance impact, certainty, and time
            scores = []
            
            for rec in recommendations:
                impact_score = 1 - (rec['expected_impact'] / 50)  # Normalize to 50 bps
                certainty_score = rec['execution_certainty']
                time_score = 1 - min(rec.get('expected_fill_time', 0) / 600, 1)  # Normalize to 10 min
                
                overall_score = (impact_score + certainty_score + time_score) / 3
                scores.append(overall_score)
            
            best_idx = np.argmax(scores)
            return recommendations[best_idx]
    
    def _aggregate_signal_sentiment(self, signals: List[Dict[str, Any]]) -> str:
        """Aggregate signals into overall sentiment"""
        
        if not signals:
            return 'neutral'
        
        bullish_weight = 0
        bearish_weight = 0
        
        for signal in signals:
            confidence = signal.get('confidence', 0.5)
            
            if signal['type'] == 'liquidity_imbalance':
                if signal['direction'] == 'bullish':
                    bullish_weight += confidence
                elif signal['direction'] == 'bearish':
                    bearish_weight += confidence
            
            elif signal['type'] == 'bid_size_dominance':
                bullish_weight += confidence * 0.5
            
            elif signal['type'] == 'ask_size_dominance':
                bearish_weight += confidence * 0.5
        
        if bullish_weight > bearish_weight * 1.2:
            return 'bullish'
        elif bearish_weight > bullish_weight * 1.2:
            return 'bearish'
        else:
            return 'neutral'
    
    def get_analyzer_statistics(self) -> Dict[str, Any]:
        """Get comprehensive analyzer statistics"""
        
        if not self.analysis_history:
            return {'status': 'no_data'}
        
        recent_analyses = list(self.analysis_history)[-100:]
        
        # Liquidity statistics
        liquidity_scores = [
            analysis['liquidity_analysis']['liquidity_score']
            for analysis in recent_analyses
        ]
        
        # Spread statistics
        spreads = [
            analysis['snapshot_data']['spread_bps']
            for analysis in recent_analyses
            if analysis['snapshot_data']['spread_bps'] is not None
        ]
        
        # Pattern statistics
        pattern_counts = [
            analysis['microstructure_patterns']['pattern_count']
            for analysis in recent_analyses
        ]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'analysis_count': len(self.analysis_history),
            'recent_analyses': len(recent_analyses),
            'liquidity_stats': {
                'avg_liquidity_score': np.mean(liquidity_scores) if liquidity_scores else 0,
                'liquidity_volatility': np.std(liquidity_scores) if len(liquidity_scores) > 1 else 0
            },
            'spread_stats': {
                'avg_spread_bps': np.mean(spreads) if spreads else 0,
                'spread_volatility': np.std(spreads) if len(spreads) > 1 else 0,
                'min_spread': np.min(spreads) if spreads else 0,
                'max_spread': np.max(spreads) if spreads else 0
            },
            'pattern_stats': {
                'avg_patterns_per_analysis': np.mean(pattern_counts) if pattern_counts else 0,
                'pattern_detection_rate': len([c for c in pattern_counts if c > 0]) / len(pattern_counts) if pattern_counts else 0
            },
            'analyzer_config': {
                'max_levels': self.config.max_levels,
                'pattern_detection_enabled': self.config.pattern_detection_enabled,
                'queue_analysis_enabled': self.config.queue_analysis_enabled
            }
        }