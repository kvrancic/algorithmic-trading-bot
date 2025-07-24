"""
Adaptive Stop-Loss Management System

Advanced stop-loss management with:
- Trailing stops with volatility adjustment
- Support/resistance level integration
- Market regime awareness
- Dynamic stop adjustment based on momentum
- Risk-parity stop allocation
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class StopLossConfig:
    """Configuration for adaptive stop-loss system"""
    
    # Basic stop-loss settings
    default_stop_pct: float = 0.05  # 5% default stop-loss
    max_stop_pct: float = 0.15  # Maximum 15% stop-loss
    min_stop_pct: float = 0.01  # Minimum 1% stop-loss
    
    # Trailing stop settings
    enable_trailing_stops: bool = True
    trailing_trigger_pct: float = 0.02  # Start trailing after 2% gain
    trailing_step_pct: float = 0.005  # Move stop by 0.5% increments
    
    # Volatility adjustment
    volatility_adjustment: bool = True
    volatility_window: int = 20  # Days for volatility calculation
    vol_multiplier: float = 2.0  # Volatility multiplier for stop distance
    
    # Technical analysis integration
    use_support_resistance: bool = True
    support_lookback: int = 50  # Days to look back for support levels
    resistance_buffer: float = 0.002  # 0.2% buffer above/below S/R levels
    
    # Market regime awareness
    regime_adjustment: bool = True
    bull_market_multiplier: float = 1.2  # Wider stops in bull markets
    bear_market_multiplier: float = 0.8  # Tighter stops in bear markets
    volatile_market_multiplier: float = 1.5  # Wider stops in volatile markets
    
    # Time-based adjustments
    time_decay_enabled: bool = True
    max_hold_days: int = 30  # Maximum holding period
    time_decay_rate: float = 0.1  # Daily tightening rate after max_hold_days
    
    # Risk management
    max_positions_at_stop: int = 3  # Max positions hitting stops per day
    stop_loss_cooldown: int = 24  # Hours before re-entering after stop
    
    def __post_init__(self):
        """Validate configuration"""
        if not (0 < self.default_stop_pct < 1):
            raise ValueError("default_stop_pct must be between 0 and 1")
        if self.max_stop_pct <= self.min_stop_pct:
            raise ValueError("max_stop_pct must be greater than min_stop_pct")


class TechnicalAnalysis:
    """Technical analysis for stop-loss optimization"""
    
    @staticmethod
    def find_support_resistance(
        prices: pd.Series,
        window: int = 20,
        min_touches: int = 2
    ) -> Dict[str, List[float]]:
        """
        Find support and resistance levels using local minima/maxima
        
        Args:
            prices: Price series
            window: Window for local extrema detection
            min_touches: Minimum touches to confirm level
            
        Returns:
            Dictionary with support and resistance levels
        """
        
        if len(prices) < window * 2:
            return {"support": [], "resistance": []}
        
        # Find local minima (support) and maxima (resistance)
        support_levels = []
        resistance_levels = []
        
        for i in range(window, len(prices) - window):
            # Check if current price is local minimum (support)
            window_prices = prices.iloc[i-window:i+window+1]
            if prices.iloc[i] == window_prices.min():
                support_levels.append(prices.iloc[i])
            
            # Check if current price is local maximum (resistance)
            if prices.iloc[i] == window_prices.max():
                resistance_levels.append(prices.iloc[i])
        
        # Cluster similar levels
        support_levels = TechnicalAnalysis._cluster_levels(support_levels, tolerance=0.02)
        resistance_levels = TechnicalAnalysis._cluster_levels(resistance_levels, tolerance=0.02)
        
        # Filter by minimum touches
        support_final = []
        resistance_final = []
        
        for level in support_levels:
            touches = sum(1 for price in prices if abs(price - level) / level < 0.01)
            if touches >= min_touches:
                support_final.append(level)
        
        for level in resistance_levels:
            touches = sum(1 for price in prices if abs(price - level) / level < 0.01)
            if touches >= min_touches:
                resistance_final.append(level)
        
        return {
            "support": sorted(support_final),
            "resistance": sorted(resistance_final, reverse=True)
        }
    
    @staticmethod
    def _cluster_levels(levels: List[float], tolerance: float = 0.02) -> List[float]:
        """Cluster similar price levels together"""
        
        if not levels:
            return []
        
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clusters.append(np.mean(current_cluster))
        return clusters
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range for volatility-based stops"""
        
        if len(high) < period:
            return pd.Series(index=close.index, data=np.nan)
        
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def detect_market_regime(returns: pd.Series, window: int = 50) -> str:
        """
        Detect market regime (bull, bear, volatile)
        
        Args:
            returns: Return series
            window: Lookback window
            
        Returns:
            Market regime string
        """
        
        if len(returns) < window:
            return "neutral"
        
        recent_returns = returns.tail(window)
        
        # Calculate metrics
        mean_return = recent_returns.mean()
        volatility = recent_returns.std()
        trend_strength = abs(mean_return) / volatility if volatility > 0 else 0
        
        # Classification thresholds
        bull_threshold = 0.001  # Daily return threshold
        bear_threshold = -0.001
        vol_threshold = 0.02  # Daily volatility threshold
        
        if mean_return > bull_threshold and trend_strength > 0.1:
            return "bull"
        elif mean_return < bear_threshold and trend_strength > 0.1:
            return "bear"
        elif volatility > vol_threshold:
            return "volatile"
        else:
            return "neutral"


class AdaptiveStopLossManager:
    """Main adaptive stop-loss management system"""
    
    def __init__(self, config: StopLossConfig):
        self.config = config
        self.position_stops: Dict[str, Dict[str, Any]] = {}
        self.stop_history: List[Dict[str, Any]] = []
        self.positions_stopped_today: int = 0
        self.last_reset_date: datetime = datetime.now().date()
        
    def calculate_optimal_stop(
        self,
        symbol: str,
        entry_price: float,
        position_size: float,
        price_data: pd.DataFrame,
        is_long: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate optimal stop-loss level for a position
        
        Args:
            symbol: Asset symbol
            entry_price: Entry price
            position_size: Position size (positive for long, negative for short)
            price_data: Historical price data (OHLCV)
            is_long: Whether position is long or short
            
        Returns:
            Stop-loss calculation result
        """
        
        logger.info(f"Calculating optimal stop for {symbol}",
                   entry_price=entry_price, position_size=position_size, is_long=is_long)
        
        result = {
            'symbol': symbol,
            'entry_price': entry_price,
            'position_size': position_size,
            'is_long': is_long,
            'timestamp': datetime.now().isoformat(),
            'stop_components': {}
        }
        
        try:
            # Base stop-loss percentage
            base_stop_pct = self.config.default_stop_pct
            result['stop_components']['base_stop'] = base_stop_pct
            
            # Volatility adjustment
            if self.config.volatility_adjustment and len(price_data) >= self.config.volatility_window:
                vol_adjustment = self._calculate_volatility_adjustment(price_data)
                base_stop_pct *= vol_adjustment
                result['stop_components']['volatility_adjustment'] = vol_adjustment
            
            # Support/resistance adjustment
            if self.config.use_support_resistance:
                sr_adjustment = self._calculate_sr_adjustment(
                    entry_price, price_data['close'], is_long
                )
                if sr_adjustment:
                    base_stop_pct = sr_adjustment
                    result['stop_components']['support_resistance_stop'] = sr_adjustment
            
            # Market regime adjustment
            if self.config.regime_adjustment and len(price_data) >= 50:
                regime_adjustment = self._calculate_regime_adjustment(price_data)
                base_stop_pct *= regime_adjustment['multiplier']
                result['stop_components']['regime_adjustment'] = regime_adjustment
            
            # Apply bounds
            final_stop_pct = np.clip(base_stop_pct, self.config.min_stop_pct, self.config.max_stop_pct)
            result['stop_components']['final_stop_pct'] = final_stop_pct
            
            # Calculate stop price
            if is_long:
                stop_price = entry_price * (1 - final_stop_pct)
            else:
                stop_price = entry_price * (1 + final_stop_pct)
            
            result['stop_price'] = stop_price
            result['stop_distance_pct'] = final_stop_pct
            result['risk_amount'] = abs(position_size * entry_price * final_stop_pct)
            
            # Store position stop
            self.position_stops[symbol] = {
                'entry_price': entry_price,
                'stop_price': stop_price,
                'stop_pct': final_stop_pct,
                'is_long': is_long,
                'entry_time': datetime.now(),
                'last_update': datetime.now(),
                'trailing_active': False,
                'highest_price': entry_price if is_long else entry_price,
                'lowest_price': entry_price if not is_long else entry_price
            }
            
            logger.info(f"Stop calculated for {symbol}",
                       stop_price=stop_price, stop_pct=final_stop_pct)
            
        except Exception as e:
            logger.error(f"Error calculating stop for {symbol}", error=str(e))
            result['error'] = str(e)
            result['stop_price'] = entry_price * (0.95 if is_long else 1.05)  # Fallback
        
        return result
    
    def update_trailing_stops(
        self,
        current_prices: Dict[str, float]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Update trailing stops for all positions
        
        Args:
            current_prices: Dictionary of current prices
            
        Returns:
            Dictionary of stop updates
        """
        
        updates = {}
        
        for symbol, stop_data in self.position_stops.items():
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            update = self._update_single_trailing_stop(symbol, current_price, stop_data)
            
            if update['updated']:
                updates[symbol] = update
                # Update stored data
                self.position_stops[symbol].update(update['new_stop_data'])
        
        return updates
    
    def _update_single_trailing_stop(
        self,
        symbol: str,
        current_price: float,
        stop_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update trailing stop for a single position"""
        
        update_result = {
            'symbol': symbol,
            'updated': False,
            'old_stop_price': stop_data['stop_price'],
            'new_stop_price': stop_data['stop_price'],
            'reason': '',
            'new_stop_data': {}
        }
        
        if not self.config.enable_trailing_stops:
            return update_result
        
        is_long = stop_data['is_long']
        entry_price = stop_data['entry_price']
        current_stop = stop_data['stop_price']
        
        # Update price extremes
        if is_long:
            new_highest = max(stop_data['highest_price'], current_price)
            update_result['new_stop_data']['highest_price'] = new_highest
            
            # Check if trailing should activate
            gain_pct = (current_price - entry_price) / entry_price
            if gain_pct >= self.config.trailing_trigger_pct:
                # Calculate new trailing stop
                trailing_distance = new_highest * self.config.trailing_step_pct
                new_stop = new_highest - trailing_distance
                
                # Only move stop up (never down for long positions)
                if new_stop > current_stop:
                    update_result['updated'] = True
                    update_result['new_stop_price'] = new_stop
                    update_result['reason'] = 'trailing_stop_adjustment'
                    update_result['new_stop_data']['stop_price'] = new_stop
                    update_result['new_stop_data']['trailing_active'] = True
        else:
            # Short position logic
            new_lowest = min(stop_data['lowest_price'], current_price)
            update_result['new_stop_data']['lowest_price'] = new_lowest
            
            gain_pct = (entry_price - current_price) / entry_price
            if gain_pct >= self.config.trailing_trigger_pct:
                trailing_distance = new_lowest * self.config.trailing_step_pct
                new_stop = new_lowest + trailing_distance
                
                # Only move stop down (never up for short positions)
                if new_stop < current_stop:
                    update_result['updated'] = True
                    update_result['new_stop_price'] = new_stop
                    update_result['reason'] = 'trailing_stop_adjustment'
                    update_result['new_stop_data']['stop_price'] = new_stop
                    update_result['new_stop_data']['trailing_active'] = True
        
        # Update last update time
        update_result['new_stop_data']['last_update'] = datetime.now()
        
        return update_result
    
    def check_stop_triggers(
        self,
        current_prices: Dict[str, float]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Check which positions should be stopped out
        
        Args:
            current_prices: Current market prices
            
        Returns:
            Dictionary of triggered stops
        """
        
        # Reset daily counter if new day
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.positions_stopped_today = 0
            self.last_reset_date = current_date
        
        triggered_stops = {}
        
        for symbol, stop_data in self.position_stops.items():
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            is_long = stop_data['is_long']
            stop_price = stop_data['stop_price']
            
            # Check stop trigger
            stop_triggered = False
            if is_long and current_price <= stop_price:
                stop_triggered = True
            elif not is_long and current_price >= stop_price:
                stop_triggered = True
            
            if stop_triggered:
                # Check daily limit
                if self.positions_stopped_today >= self.config.max_positions_at_stop:
                    logger.warning(f"Daily stop limit reached, skipping {symbol}")
                    continue
                
                triggered_stops[symbol] = {
                    'symbol': symbol,
                    'entry_price': stop_data['entry_price'],
                    'stop_price': stop_price,
                    'current_price': current_price,
                    'is_long': is_long,
                    'pnl_pct': self._calculate_pnl_pct(
                        stop_data['entry_price'], current_price, is_long
                    ),
                    'trigger_time': datetime.now().isoformat(),
                    'hold_duration': datetime.now() - stop_data['entry_time']
                }
                
                self.positions_stopped_today += 1
                
                # Record in history
                self.stop_history.append(triggered_stops[symbol])
                
                # Remove from active stops
                del self.position_stops[symbol]
        
        return triggered_stops
    
    def _calculate_volatility_adjustment(self, price_data: pd.DataFrame) -> float:
        """Calculate volatility-based stop adjustment"""
        
        returns = price_data['close'].pct_change().dropna()
        if len(returns) < self.config.volatility_window:
            return 1.0
        
        recent_returns = returns.tail(self.config.volatility_window)
        volatility = recent_returns.std()
        
        # Use ATR if available
        if all(col in price_data.columns for col in ['high', 'low', 'close']):
            atr = TechnicalAnalysis.calculate_atr(
                price_data['high'], price_data['low'], price_data['close']
            )
            if not atr.empty:
                recent_atr = atr.dropna().tail(1).iloc[0]
                current_price = price_data['close'].iloc[-1]
                atr_pct = recent_atr / current_price
                return min(3.0, max(0.5, atr_pct * self.config.vol_multiplier / self.config.default_stop_pct))
        
        # Fallback to return volatility
        return min(3.0, max(0.5, volatility * self.config.vol_multiplier / 0.01))
    
    def _calculate_sr_adjustment(
        self,
        entry_price: float,
        price_series: pd.Series,
        is_long: bool
    ) -> Optional[float]:
        """Calculate support/resistance based stop adjustment"""
        
        sr_levels = TechnicalAnalysis.find_support_resistance(
            price_series, self.config.support_lookback
        )
        
        if is_long:
            # Find nearest support level below entry
            relevant_supports = [s for s in sr_levels['support'] if s < entry_price]
            if relevant_supports:
                nearest_support = max(relevant_supports)
                buffer_price = nearest_support * (1 - self.config.resistance_buffer)
                stop_pct = (entry_price - buffer_price) / entry_price
                return max(self.config.min_stop_pct, min(self.config.max_stop_pct, stop_pct))
        else:
            # Find nearest resistance level above entry
            relevant_resistance = [r for r in sr_levels['resistance'] if r > entry_price]
            if relevant_resistance:
                nearest_resistance = min(relevant_resistance)
                buffer_price = nearest_resistance * (1 + self.config.resistance_buffer)
                stop_pct = (buffer_price - entry_price) / entry_price
                return max(self.config.min_stop_pct, min(self.config.max_stop_pct, stop_pct))
        
        return None
    
    def _calculate_regime_adjustment(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate market regime based stop adjustment"""
        
        returns = price_data['close'].pct_change().dropna()
        regime = TechnicalAnalysis.detect_market_regime(returns)
        
        regime_multipliers = {
            'bull': self.config.bull_market_multiplier,
            'bear': self.config.bear_market_multiplier,
            'volatile': self.config.volatile_market_multiplier,
            'neutral': 1.0
        }
        
        return {
            'regime': regime,
            'multiplier': regime_multipliers.get(regime, 1.0)
        }
    
    def _calculate_pnl_pct(self, entry_price: float, exit_price: float, is_long: bool) -> float:
        """Calculate P&L percentage"""
        if is_long:
            return (exit_price - entry_price) / entry_price
        else:
            return (entry_price - exit_price) / entry_price
    
    def get_stop_summary(self) -> Dict[str, Any]:
        """Get summary of all active stops"""
        
        return {
            'active_stops': len(self.position_stops),
            'positions_stopped_today': self.positions_stopped_today,
            'total_stops_history': len(self.stop_history),
            'config': {
                'default_stop_pct': self.config.default_stop_pct,
                'trailing_enabled': self.config.enable_trailing_stops,
                'volatility_adjustment': self.config.volatility_adjustment,
                'regime_adjustment': self.config.regime_adjustment
            },
            'active_positions': {
                symbol: {
                    'entry_price': data['entry_price'],
                    'stop_price': data['stop_price'],
                    'stop_pct': data['stop_pct'],
                    'is_long': data['is_long'],
                    'trailing_active': data.get('trailing_active', False)
                }
                for symbol, data in self.position_stops.items()
            }
        }
    
    def remove_position_stop(self, symbol: str) -> bool:
        """Remove stop for a position (e.g., when position is closed normally)"""
        
        if symbol in self.position_stops:
            del self.position_stops[symbol]
            logger.info(f"Removed stop for {symbol}")
            return True
        return False