"""
Maximum Drawdown Controller

Advanced drawdown management system with:
- Real-time drawdown monitoring
- Dynamic position sizing reduction
- Circuit breaker mechanisms
- Recovery protocols
- Historical drawdown analysis
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
class DrawdownConfig:
    """Configuration for drawdown controller"""
    
    # Drawdown limits
    max_daily_drawdown: float = 0.02  # 2% maximum daily drawdown
    max_weekly_drawdown: float = 0.05  # 5% maximum weekly drawdown
    max_monthly_drawdown: float = 0.10  # 10% maximum monthly drawdown
    max_total_drawdown: float = 0.20  # 20% maximum total drawdown
    
    # Circuit breaker levels
    warning_drawdown: float = 0.01  # 1% warning level
    caution_drawdown: float = 0.015  # 1.5% caution level
    critical_drawdown: float = 0.018  # 1.8% critical level
    
    # Position sizing adjustments
    enable_dynamic_sizing: bool = True
    max_size_reduction: float = 0.8  # Reduce position size by up to 80%
    size_reduction_threshold: float = 0.01  # Start reducing at 1% drawdown
    
    # Recovery settings
    recovery_threshold: float = 0.5  # Recover 50% of drawdown before resuming
    recovery_lookback: int = 5  # Days to confirm recovery
    enable_gradual_recovery: bool = True
    
    # Risk-off settings
    risk_off_threshold: float = 0.015  # Go risk-off at 1.5% drawdown
    risk_off_duration: int = 24  # Hours to remain risk-off
    
    # Monitoring settings
    update_frequency: int = 15  # Minutes between updates
    high_water_mark_reset: int = 30  # Days to reset high water mark
    
    def __post_init__(self):
        """Validate configuration"""
        drawdowns = [
            self.max_daily_drawdown,
            self.max_weekly_drawdown, 
            self.max_monthly_drawdown,
            self.max_total_drawdown
        ]
        
        if not all(0 < dd < 1 for dd in drawdowns):
            raise ValueError("All drawdown limits must be between 0 and 1")
            
        if not (self.max_daily_drawdown <= self.max_weekly_drawdown <= 
                self.max_monthly_drawdown <= self.max_total_drawdown):
            raise ValueError("Drawdown limits should be progressive (daily <= weekly <= monthly <= total)")


class DrawdownCalculator:
    """Calculate various drawdown metrics"""
    
    @staticmethod
    def calculate_drawdown_series(equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series from equity curve"""
        
        if len(equity_curve) == 0:
            return pd.Series(dtype=float)
        
        # Calculate running maximum (high water mark)
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown as percentage
        drawdown = (equity_curve - running_max) / running_max
        
        return drawdown
    
    @staticmethod
    def calculate_drawdown_periods(drawdown_series: pd.Series) -> List[Dict[str, Any]]:
        """Identify and analyze drawdown periods"""
        
        periods = []
        in_drawdown = False
        start_idx = None
        start_date = None
        peak_value = 0
        trough_value = 0
        
        for idx, (date, dd_value) in enumerate(drawdown_series.items()):
            if dd_value < 0 and not in_drawdown:
                # Start of drawdown period
                in_drawdown = True
                start_idx = idx
                start_date = date
                peak_value = 0  # High water mark
                trough_value = dd_value
                
            elif dd_value < 0 and in_drawdown:
                # Continue drawdown period
                trough_value = min(trough_value, dd_value)
                
            elif dd_value >= 0 and in_drawdown:
                # End of drawdown period
                in_drawdown = False
                
                periods.append({
                    'start_date': start_date,
                    'end_date': date,
                    'duration_days': (date - start_date).days,
                    'max_drawdown': trough_value,
                    'start_idx': start_idx,
                    'end_idx': idx
                })
        
        # Handle ongoing drawdown
        if in_drawdown and start_date is not None:
            periods.append({
                'start_date': start_date,
                'end_date': drawdown_series.index[-1],
                'duration_days': (drawdown_series.index[-1] - start_date).days,
                'max_drawdown': trough_value,
                'start_idx': start_idx,
                'end_idx': len(drawdown_series) - 1,
                'ongoing': True
            })
        
        return periods
    
    @staticmethod
    def calculate_recovery_metrics(
        equity_curve: pd.Series,
        drawdown_periods: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate recovery time and factor metrics"""
        
        recovery_times = []
        recovery_factors = []
        
        for period in drawdown_periods:
            if period.get('ongoing', False):
                continue
                
            start_idx = period['start_idx']
            end_idx = period['end_idx']
            
            # Find recovery point (when equity reaches new high)
            recovery_idx = None
            start_value = equity_curve.iloc[start_idx]
            
            for i in range(end_idx, len(equity_curve)):
                if equity_curve.iloc[i] >= start_value:
                    recovery_idx = i
                    break
            
            if recovery_idx is not None:
                recovery_days = (equity_curve.index[recovery_idx] - 
                               equity_curve.index[start_idx]).days
                recovery_times.append(recovery_days)
                
                # Recovery factor: gain needed to recover from drawdown
                trough_value = equity_curve.iloc[start_idx] * (1 + period['max_drawdown'])
                recovery_factor = (start_value - trough_value) / trough_value
                recovery_factors.append(recovery_factor)
        
        return {
            'avg_recovery_days': np.mean(recovery_times) if recovery_times else 0,
            'max_recovery_days': max(recovery_times) if recovery_times else 0,
            'avg_recovery_factor': np.mean(recovery_factors) if recovery_factors else 0,
            'recovery_success_rate': len(recovery_times) / len(drawdown_periods) if drawdown_periods else 0
        }


class RiskOffManager:
    """Manage risk-off periods during excessive drawdowns"""
    
    def __init__(self, config: DrawdownConfig):
        self.config = config
        self.risk_off_start: Optional[datetime] = None
        self.risk_off_reason: str = ""
        self.pre_risk_off_positions: Dict[str, float] = {}
        
    def should_go_risk_off(
        self,
        current_drawdown: float,
        drawdown_velocity: float = 0.0
    ) -> Tuple[bool, str]:
        """
        Determine if system should go risk-off
        
        Args:
            current_drawdown: Current drawdown percentage
            drawdown_velocity: Rate of drawdown increase
            
        Returns:
            Tuple of (should_go_risk_off, reason)
        """
        
        # Already risk-off
        if self.is_risk_off():
            return True, self.risk_off_reason
        
        # Check drawdown threshold
        if abs(current_drawdown) >= self.config.risk_off_threshold:
            return True, f"Drawdown {current_drawdown:.2%} exceeds threshold {self.config.risk_off_threshold:.2%}"
        
        # Check velocity-based trigger
        if drawdown_velocity < -0.005:  # Rapid drawdown
            return True, f"Rapid drawdown velocity: {drawdown_velocity:.3%}/period"
        
        return False, ""
    
    def enter_risk_off(
        self,
        reason: str,
        current_positions: Dict[str, float]
    ) -> Dict[str, Any]:
        """Enter risk-off mode"""
        
        if self.is_risk_off():
            return {'status': 'already_risk_off', 'start_time': self.risk_off_start}
        
        self.risk_off_start = datetime.now()
        self.risk_off_reason = reason
        self.pre_risk_off_positions = current_positions.copy()
        
        logger.critical("Entering RISK-OFF mode", reason=reason)
        
        return {
            'status': 'risk_off_activated',
            'start_time': self.risk_off_start,
            'reason': reason,
            'positions_frozen': len(current_positions),
            'expected_duration_hours': self.config.risk_off_duration
        }
    
    def should_exit_risk_off(
        self,
        current_drawdown: float,
        recovery_confirmed: bool = False
    ) -> Tuple[bool, str]:
        """
        Determine if system should exit risk-off mode
        
        Args:
            current_drawdown: Current drawdown
            recovery_confirmed: Whether recovery is confirmed
            
        Returns:
            Tuple of (should_exit, reason)
        """
        
        if not self.is_risk_off():
            return False, "Not in risk-off mode"
        
        # Check minimum duration
        time_in_risk_off = datetime.now() - self.risk_off_start
        if time_in_risk_off.total_seconds() < self.config.risk_off_duration * 3600:
            return False, f"Minimum risk-off duration not met: {time_in_risk_off}"
        
        # Check recovery condition
        recovery_threshold = self.config.recovery_threshold
        if abs(current_drawdown) <= self.config.risk_off_threshold * recovery_threshold:
            if recovery_confirmed:
                return True, f"Drawdown recovered to {current_drawdown:.2%}"
            else:
                return False, "Recovery not yet confirmed"
        
        return False, f"Drawdown {current_drawdown:.2%} still above recovery threshold"
    
    def exit_risk_off(self) -> Dict[str, Any]:
        """Exit risk-off mode"""
        
        if not self.is_risk_off():
            return {'status': 'not_risk_off'}
        
        duration = datetime.now() - self.risk_off_start
        
        result = {
            'status': 'risk_off_deactivated',
            'start_time': self.risk_off_start,
            'end_time': datetime.now(),
            'duration': duration,
            'reason': self.risk_off_reason,
            'pre_risk_off_positions': self.pre_risk_off_positions
        }
        
        # Reset state
        self.risk_off_start = None
        self.risk_off_reason = ""
        self.pre_risk_off_positions = {}
        
        logger.info("Exiting RISK-OFF mode", duration_hours=duration.total_seconds()/3600)
        
        return result
    
    def is_risk_off(self) -> bool:
        """Check if currently in risk-off mode"""
        return self.risk_off_start is not None


class DrawdownController:
    """Main drawdown control system"""
    
    def __init__(self, config: DrawdownConfig):
        self.config = config
        self.calculator = DrawdownCalculator()
        self.risk_off_manager = RiskOffManager(config)
        
        # State tracking
        self.high_water_mark: float = 0.0
        self.current_equity: float = 0.0
        self.equity_history: List[Dict[str, Any]] = []
        self.drawdown_alerts: List[Dict[str, Any]] = []
        
        # Dynamic sizing
        self.current_size_multiplier: float = 1.0
        self.last_update: datetime = datetime.now()
        
    def update_equity(self, current_equity: float) -> Dict[str, Any]:
        """
        Update current equity and perform drawdown analysis
        
        Args:
            current_equity: Current portfolio equity
            
        Returns:
            Drawdown analysis and control decisions
        """
        
        self.current_equity = current_equity
        self.last_update = datetime.now()
        
        # Update high water mark
        if current_equity > self.high_water_mark:
            self.high_water_mark = current_equity
        
        # Record equity point
        equity_point = {
            'timestamp': self.last_update,
            'equity': current_equity,
            'high_water_mark': self.high_water_mark
        }
        self.equity_history.append(equity_point)
        
        # Keep only recent history (configurable window)
        max_history_days = max(30, self.config.high_water_mark_reset)
        cutoff_date = self.last_update - timedelta(days=max_history_days)
        self.equity_history = [
            point for point in self.equity_history 
            if point['timestamp'] > cutoff_date
        ]
        
        # Calculate current drawdown
        current_drawdown = (current_equity - self.high_water_mark) / self.high_water_mark
        
        # Perform analysis
        analysis = self._analyze_current_state(current_drawdown)
        
        # Make control decisions
        control_actions = self._make_control_decisions(analysis)
        
        return {
            'timestamp': self.last_update.isoformat(),
            'current_equity': current_equity,
            'high_water_mark': self.high_water_mark,
            'current_drawdown': current_drawdown,
            'analysis': analysis,
            'control_actions': control_actions,
            'risk_off_status': self.risk_off_manager.is_risk_off()
        }
    
    def _analyze_current_state(self, current_drawdown: float) -> Dict[str, Any]:
        """Analyze current drawdown state"""
        
        # Convert equity history to series for analysis
        if len(self.equity_history) < 2:
            return {
                'current_drawdown': current_drawdown,
                'drawdown_velocity': 0.0,
                'time_periods': {},
                'severity_level': 'normal'
            }
        
        equity_series = pd.Series(
            data=[point['equity'] for point in self.equity_history],
            index=[point['timestamp'] for point in self.equity_history]
        )
        
        # Calculate drawdown series
        drawdown_series = self.calculator.calculate_drawdown_series(equity_series)
        
        # Calculate velocity (rate of change)
        if len(drawdown_series) >= 2:
            recent_change = drawdown_series.iloc[-1] - drawdown_series.iloc[-2]
            time_diff = (drawdown_series.index[-1] - drawdown_series.index[-2]).total_seconds() / 3600
            drawdown_velocity = recent_change / time_diff if time_diff > 0 else 0
        else:
            drawdown_velocity = 0.0
        
        # Analyze time-based drawdowns
        time_periods = self._analyze_time_periods(equity_series)
        
        # Determine severity level
        severity_level = self._determine_severity_level(current_drawdown)
        
        return {
            'current_drawdown': current_drawdown,
            'drawdown_velocity': drawdown_velocity,
            'time_periods': time_periods,
            'severity_level': severity_level,
            'in_drawdown': current_drawdown < 0,
            'drawdown_duration': self._calculate_current_drawdown_duration(drawdown_series)
        }
    
    def _analyze_time_periods(self, equity_series: pd.Series) -> Dict[str, Dict[str, float]]:
        """Analyze drawdowns over different time periods"""
        
        now = datetime.now()
        periods = {
            'daily': timedelta(days=1),
            'weekly': timedelta(weeks=1),
            'monthly': timedelta(days=30)
        }
        
        results = {}
        
        for period_name, period_duration in periods.items():
            period_start = now - period_duration
            period_data = equity_series[equity_series.index >= period_start]
            
            if len(period_data) >= 2:
                period_high = period_data.max()
                period_current = period_data.iloc[-1]
                period_drawdown = (period_current - period_high) / period_high
                
                results[period_name] = {
                    'drawdown': period_drawdown,
                    'high': period_high,
                    'current': period_current,
                    'data_points': len(period_data)
                }
            else:
                results[period_name] = {
                    'drawdown': 0.0,
                    'high': self.current_equity,
                    'current': self.current_equity,
                    'data_points': len(period_data)
                }
        
        return results
    
    def _determine_severity_level(self, current_drawdown: float) -> str:
        """Determine drawdown severity level"""
        
        abs_drawdown = abs(current_drawdown)
        
        if abs_drawdown >= self.config.critical_drawdown:
            return 'critical'
        elif abs_drawdown >= self.config.caution_drawdown:
            return 'caution'
        elif abs_drawdown >= self.config.warning_drawdown:
            return 'warning'
        else:
            return 'normal'
    
    def _calculate_current_drawdown_duration(self, drawdown_series: pd.Series) -> Optional[timedelta]:
        """Calculate how long current drawdown has lasted"""
        
        if len(drawdown_series) == 0 or drawdown_series.iloc[-1] >= 0:
            return None
        
        # Find when current drawdown period started
        for i in range(len(drawdown_series) - 1, -1, -1):
            if drawdown_series.iloc[i] >= 0:
                if i < len(drawdown_series) - 1:
                    start_time = drawdown_series.index[i + 1]
                    return datetime.now() - start_time.to_pydatetime()
                break
        
        # If no zero found, drawdown started at beginning of data
        if len(drawdown_series) > 0:
            return datetime.now() - drawdown_series.index[0].to_pydatetime()
        
        return None
    
    def _make_control_decisions(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make control decisions based on analysis"""
        
        actions = {
            'alerts': [],
            'position_size_adjustment': 1.0,
            'risk_off_decision': None,
            'recovery_status': None,
            'circuit_breaker': False
        }
        
        current_drawdown = analysis['current_drawdown']
        severity_level = analysis['severity_level']
        time_periods = analysis['time_periods']
        
        # Generate alerts
        actions['alerts'] = self._generate_drawdown_alerts(analysis)
        
        # Position sizing adjustment
        if self.config.enable_dynamic_sizing:
            actions['position_size_adjustment'] = self._calculate_position_size_adjustment(
                current_drawdown
            )
        
        # Risk-off decision
        should_risk_off, risk_off_reason = self.risk_off_manager.should_go_risk_off(
            current_drawdown, analysis['drawdown_velocity']
        )
        
        if should_risk_off and not self.risk_off_manager.is_risk_off():
            actions['risk_off_decision'] = {
                'action': 'enter_risk_off',
                'reason': risk_off_reason
            }
        elif self.risk_off_manager.is_risk_off():
            should_exit, exit_reason = self.risk_off_manager.should_exit_risk_off(
                current_drawdown, recovery_confirmed=True  # Simplified
            )
            if should_exit:
                actions['risk_off_decision'] = {
                    'action': 'exit_risk_off',
                    'reason': exit_reason
                }
        
        # Circuit breaker
        max_limits = [
            self.config.max_daily_drawdown,
            self.config.max_weekly_drawdown,
            self.config.max_monthly_drawdown,
            self.config.max_total_drawdown
        ]
        
        current_drawdowns = [
            abs(time_periods.get('daily', {}).get('drawdown', 0)),
            abs(time_periods.get('weekly', {}).get('drawdown', 0)),
            abs(time_periods.get('monthly', {}).get('drawdown', 0)),
            abs(current_drawdown)
        ]
        
        for i, (current_dd, max_dd) in enumerate(zip(current_drawdowns, max_limits)):
            if current_dd >= max_dd:
                actions['circuit_breaker'] = True
                actions['alerts'].append({
                    'severity': 'CRITICAL',
                    'type': 'CIRCUIT_BREAKER',
                    'message': f"Circuit breaker triggered - {['daily', 'weekly', 'monthly', 'total'][i]} drawdown limit exceeded",
                    'drawdown': current_dd,
                    'limit': max_dd
                })
                break
        
        return actions
    
    def _generate_drawdown_alerts(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate appropriate alerts based on drawdown analysis"""
        
        alerts = []
        current_drawdown = analysis['current_drawdown']
        severity_level = analysis['severity_level']
        
        if severity_level == 'critical':
            alerts.append({
                'severity': 'CRITICAL',
                'type': 'CRITICAL_DRAWDOWN',
                'message': f"Critical drawdown level reached: {current_drawdown:.2%}",
                'drawdown': current_drawdown,
                'threshold': self.config.critical_drawdown
            })
        elif severity_level == 'caution':
            alerts.append({
                'severity': 'HIGH',
                'type': 'CAUTION_DRAWDOWN',
                'message': f"Caution drawdown level reached: {current_drawdown:.2%}",
                'drawdown': current_drawdown,
                'threshold': self.config.caution_drawdown
            })
        elif severity_level == 'warning':
            alerts.append({
                'severity': 'MEDIUM',
                'type': 'WARNING_DRAWDOWN',
                'message': f"Warning drawdown level reached: {current_drawdown:.2%}",
                'drawdown': current_drawdown,
                'threshold': self.config.warning_drawdown
            })
        
        # Velocity alerts
        velocity = analysis['drawdown_velocity']
        if velocity < -0.01:  # Rapid drawdown
            alerts.append({
                'severity': 'HIGH',
                'type': 'RAPID_DRAWDOWN',
                'message': f"Rapid drawdown detected: {velocity:.3%}/hour",
                'velocity': velocity
            })
        
        return alerts
    
    def _calculate_position_size_adjustment(self, current_drawdown: float) -> float:
        """Calculate position size adjustment factor"""
        
        if abs(current_drawdown) <= self.config.size_reduction_threshold:
            self.current_size_multiplier = 1.0
            return 1.0
        
        # Linear reduction based on drawdown
        excess_drawdown = abs(current_drawdown) - self.config.size_reduction_threshold
        max_excess = self.config.max_total_drawdown - self.config.size_reduction_threshold
        
        if max_excess <= 0:
            reduction_factor = 0
        else:
            reduction_factor = min(1.0, excess_drawdown / max_excess)
        
        # Calculate new multiplier
        new_multiplier = 1.0 - (reduction_factor * self.config.max_size_reduction)
        self.current_size_multiplier = max(0.1, new_multiplier)  # Minimum 10% of normal size
        
        return self.current_size_multiplier
    
    def get_drawdown_dashboard(self) -> Dict[str, Any]:
        """Get drawdown monitoring dashboard"""
        
        if not self.equity_history:
            return {'status': 'no_data'}
        
        # Calculate comprehensive metrics
        equity_series = pd.Series(
            data=[point['equity'] for point in self.equity_history],
            index=[point['timestamp'] for point in self.equity_history]
        )
        
        drawdown_series = self.calculator.calculate_drawdown_series(equity_series)
        drawdown_periods = self.calculator.calculate_drawdown_periods(drawdown_series)
        recovery_metrics = self.calculator.calculate_recovery_metrics(equity_series, drawdown_periods)
        
        current_drawdown = (self.current_equity - self.high_water_mark) / self.high_water_mark
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_status': {
                'equity': self.current_equity,
                'high_water_mark': self.high_water_mark,
                'current_drawdown': current_drawdown,
                'severity_level': self._determine_severity_level(current_drawdown),
                'risk_off': self.risk_off_manager.is_risk_off(),
                'position_size_multiplier': self.current_size_multiplier
            },
            'limits': {
                'daily': self.config.max_daily_drawdown,
                'weekly': self.config.max_weekly_drawdown,
                'monthly': self.config.max_monthly_drawdown,
                'total': self.config.max_total_drawdown
            },
            'historical_metrics': {
                'total_drawdown_periods': len(drawdown_periods),
                'max_historical_drawdown': min([p['max_drawdown'] for p in drawdown_periods]) if drawdown_periods else 0,
                'avg_drawdown_duration': np.mean([p['duration_days'] for p in drawdown_periods]) if drawdown_periods else 0,
                'recovery_metrics': recovery_metrics
            },
            'recent_alerts': self.drawdown_alerts[-10:],
            'config': {
                'dynamic_sizing_enabled': self.config.enable_dynamic_sizing,
                'circuit_breakers_enabled': True,  # Always enabled
                'risk_off_threshold': self.config.risk_off_threshold
            }
        }