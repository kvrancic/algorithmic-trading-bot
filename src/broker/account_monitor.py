"""
Account Monitoring System

Comprehensive account oversight with:
- Real-time account metrics tracking
- Risk monitoring and alerts
- Buying power and margin management
- Performance analytics
- Compliance and regulatory monitoring
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from collections import deque
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class AccountStatus(Enum):
    """Account status enumeration"""
    ACTIVE = "active"
    RESTRICTED = "restricted"
    SUSPENDED = "suspended"
    CLOSED = "closed"
    MAINTENANCE = "maintenance"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class AccountAlert:
    """Account alert representation"""
    alert_id: str
    alert_type: str
    level: AlertLevel
    message: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    
    def acknowledge(self) -> None:
        """Acknowledge the alert"""
        self.acknowledged = True
        logger.info("Alert acknowledged", alert_id=self.alert_id)
    
    def resolve(self) -> None:
        """Mark alert as resolved"""
        self.resolved = True
        logger.info("Alert resolved", alert_id=self.alert_id)


@dataclass
class AccountSnapshot:
    """Point-in-time account snapshot"""
    timestamp: datetime
    equity: float
    buying_power: float
    cash: float
    portfolio_value: float
    day_trade_count: int
    maintenance_margin: float
    initial_margin: float
    unrealized_pnl: float
    realized_pnl: float
    total_fees: float
    position_count: int
    
    @property
    def net_liquidity(self) -> float:
        """Calculate net account liquidity"""
        return self.equity - self.maintenance_margin
    
    @property
    def margin_utilization(self) -> float:
        """Calculate margin utilization percentage"""
        if self.initial_margin == 0:
            return 0.0
        return (self.portfolio_value - self.cash) / self.initial_margin * 100


@dataclass
class AccountMonitorConfig:
    """Configuration for account monitor"""
    
    # Monitoring intervals
    snapshot_interval_seconds: int = 30
    alert_check_interval_seconds: int = 5
    performance_calculation_interval: int = 300  # 5 minutes
    
    # Risk thresholds
    min_buying_power_threshold: float = 1000.0
    max_margin_utilization: float = 0.8  # 80%
    max_position_concentration: float = 0.1  # 10%
    max_daily_loss_threshold: float = 0.05  # 5%
    max_drawdown_threshold: float = 0.1  # 10%
    
    # Day trading limits
    max_day_trades_per_period: int = 3
    day_trade_period_days: int = 5
    pdt_equity_minimum: float = 25000.0  # Pattern Day Trader minimum
    
    # Performance tracking
    performance_history_days: int = 30
    benchmark_symbol: str = "SPY"
    
    # Alert settings
    enable_email_alerts: bool = False
    enable_sms_alerts: bool = False
    alert_cooldown_minutes: int = 15  # Prevent alert spam
    
    # Data retention
    snapshot_retention_days: int = 90
    alert_retention_days: int = 365
    
    def __post_init__(self):
        """Validate configuration"""
        if self.snapshot_interval_seconds <= 0:
            raise ValueError("snapshot_interval_seconds must be positive")
        if not (0 < self.max_margin_utilization <= 1):
            raise ValueError("max_margin_utilization must be between 0 and 1")


class AccountMonitor:
    """Comprehensive account monitoring system"""
    
    def __init__(self, config: AccountMonitorConfig):
        self.config = config
        
        # Account data storage
        self.current_snapshot: Optional[AccountSnapshot] = None
        self.snapshot_history: deque = deque(maxlen=1000)
        self.alerts: List[AccountAlert] = []
        self.alert_history: List[AccountAlert] = []
        
        # Performance tracking
        self.daily_pnl_history: deque = deque(maxlen=self.config.performance_history_days)
        self.equity_curve: deque = deque(maxlen=1000)
        self.benchmark_data: Dict[str, float] = {}
        
        # Alert tracking
        self.last_alert_times: Dict[str, datetime] = {}
        self.alert_callbacks: List[Callable] = []
        
        # Integration components
        self.broker = None
        self.position_tracker = None
        self.risk_engine = None
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_tasks: List[asyncio.Task] = []
        
        logger.info("Account monitor initialized", config=config)
    
    def set_broker(self, broker) -> None:
        """Set broker integration"""
        self.broker = broker
        logger.info("Broker integration set for account monitor")
    
    def set_position_tracker(self, position_tracker) -> None:
        """Set position tracker"""
        self.position_tracker = position_tracker
        logger.info("Position tracker set for account monitor")
    
    def set_risk_engine(self, risk_engine) -> None:
        """Set risk engine"""
        self.risk_engine = risk_engine
        logger.info("Risk engine set for account monitor")
    
    def add_alert_callback(self, callback: Callable[[AccountAlert], None]) -> None:
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
        logger.info("Alert callback added")
    
    async def start_monitoring(self) -> None:
        """Start account monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # Start monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._snapshot_loop()),
            asyncio.create_task(self._alert_monitoring_loop()),
            asyncio.create_task(self._performance_calculation_loop())
        ]
        
        logger.info("Account monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop account monitoring"""
        self.monitoring_active = False
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        self.monitoring_tasks.clear()
        
        logger.info("Account monitoring stopped")
    
    async def take_snapshot(self) -> Optional[AccountSnapshot]:
        """Take current account snapshot"""
        if not self.broker:
            logger.warning("No broker configured for account snapshot")
            return None
        
        try:
            # Get account info from broker
            account = await self.broker.get_account()
            
            # Get position data
            position_count = 0
            total_unrealized_pnl = 0.0
            if self.position_tracker:
                positions = self.position_tracker.get_all_positions()
                position_count = len([p for p in positions if not p.is_flat])
                total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
            
            # Create snapshot
            snapshot = AccountSnapshot(
                timestamp=datetime.now(),
                equity=float(account.equity),
                buying_power=float(account.buying_power),
                cash=float(account.cash),
                portfolio_value=float(account.portfolio_value),
                day_trade_count=int(getattr(account, 'daytrade_count', 0)),
                maintenance_margin=float(getattr(account, 'maintenance_margin', 0)),
                initial_margin=float(getattr(account, 'initial_margin', 0)),
                unrealized_pnl=total_unrealized_pnl,
                realized_pnl=0.0,  # Would need to calculate from trades
                total_fees=0.0,    # Would need to calculate from trades
                position_count=position_count
            )
            
            # Store snapshot
            self.current_snapshot = snapshot
            self.snapshot_history.append(snapshot)
            self.equity_curve.append((snapshot.timestamp, snapshot.equity))
            
            logger.debug("Account snapshot taken",
                        equity=snapshot.equity,
                        buying_power=snapshot.buying_power,
                        positions=snapshot.position_count)
            
            return snapshot
            
        except Exception as e:
            logger.error("Failed to take account snapshot", error=str(e))
            return None
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current account status"""
        if not self.current_snapshot:
            return {'status': 'no_data'}
        
        snapshot = self.current_snapshot
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics()
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics()
        
        # Get recent alerts
        recent_alerts = [
            {
                'alert_id': alert.alert_id,
                'type': alert.alert_type,
                'level': alert.level.value,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'acknowledged': alert.acknowledged,
                'resolved': alert.resolved
            }
            for alert in self.alerts[-10:]  # Last 10 alerts
        ]
        
        return {
            'timestamp': snapshot.timestamp.isoformat(),
            'account_status': 'active',  # Would determine from broker
            'equity': snapshot.equity,
            'buying_power': snapshot.buying_power,
            'cash': snapshot.cash,
            'portfolio_value': snapshot.portfolio_value,
            'net_liquidity': snapshot.net_liquidity,
            'margin_utilization': snapshot.margin_utilization,
            'day_trade_count': snapshot.day_trade_count,
            'position_count': snapshot.position_count,
            'unrealized_pnl': snapshot.unrealized_pnl,
            'performance_metrics': performance_metrics,
            'risk_metrics': risk_metrics,
            'recent_alerts': recent_alerts,
            'active_alerts': len([a for a in self.alerts if not a.resolved])
        }
    
    def get_historical_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get historical account summary"""
        if not self.snapshot_history:
            return {'status': 'no_data'}
        
        # Filter snapshots by date range
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_snapshots = [
            s for s in self.snapshot_history 
            if s.timestamp > cutoff_date
        ]
        
        if not recent_snapshots:
            return {'status': 'insufficient_data'}
        
        # Calculate historical metrics
        start_equity = recent_snapshots[0].equity
        end_equity = recent_snapshots[-1].equity
        total_return = (end_equity - start_equity) / start_equity * 100 if start_equity > 0 else 0
        
        # Equity statistics
        equities = [s.equity for s in recent_snapshots]
        max_equity = max(equities)
        min_equity = min(equities)
        avg_equity = np.mean(equities)
        equity_volatility = np.std(equities) / avg_equity * 100 if avg_equity > 0 else 0
        
        # Drawdown calculation
        peak_equity = start_equity
        max_drawdown = 0.0
        for snapshot in recent_snapshots:
            if snapshot.equity > peak_equity:
                peak_equity = snapshot.equity
            drawdown = (peak_equity - snapshot.equity) / peak_equity * 100 if peak_equity > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        # Trading activity
        avg_positions = np.mean([s.position_count for s in recent_snapshots])
        total_day_trades = sum(s.day_trade_count for s in recent_snapshots)
        
        return {
            'period_days': days,
            'start_date': recent_snapshots[0].timestamp.isoformat(),
            'end_date': recent_snapshots[-1].timestamp.isoformat(),
            'start_equity': start_equity,
            'end_equity': end_equity,
            'total_return_percent': total_return,
            'max_equity': max_equity,
            'min_equity': min_equity,
            'avg_equity': avg_equity,
            'equity_volatility_percent': equity_volatility,
            'max_drawdown_percent': max_drawdown,
            'avg_positions': avg_positions,
            'total_day_trades': total_day_trades,
            'snapshots_count': len(recent_snapshots)
        }
    
    async def create_alert(
        self,
        alert_type: str,
        level: AlertLevel,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> AccountAlert:
        """
        Create and process new alert
        
        Args:
            alert_type: Type of alert
            level: Alert severity level
            message: Alert message
            data: Additional alert data
            
        Returns:
            Created alert
        """
        
        # Check alert cooldown
        cooldown_key = f"{alert_type}_{level.value}"
        last_alert_time = self.last_alert_times.get(cooldown_key)
        if last_alert_time:
            time_since_last = datetime.now() - last_alert_time
            if time_since_last < timedelta(minutes=self.config.alert_cooldown_minutes):
                logger.debug("Alert skipped due to cooldown",
                           alert_type=alert_type,
                           level=level.value)
                return None
        
        # Create alert
        alert = AccountAlert(
            alert_id=f"{alert_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            alert_type=alert_type,
            level=level,
            message=message,
            timestamp=datetime.now(),
            data=data or {}
        )
        
        # Store alert
        self.alerts.append(alert)
        self.last_alert_times[cooldown_key] = alert.timestamp
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error("Alert callback error",
                           alert_id=alert.alert_id,
                           error=str(e))
        
        logger.info("Account alert created",
                   alert_id=alert.alert_id,
                   type=alert_type,
                   level=level.value,
                   message=message)
        
        return alert
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledge()
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolve()
                return True
        return False
    
    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        resolved: Optional[bool] = None,
        limit: int = 100
    ) -> List[AccountAlert]:
        """Get alerts with optional filtering"""
        filtered_alerts = self.alerts
        
        if level is not None:
            filtered_alerts = [a for a in filtered_alerts if a.level == level]
        
        if resolved is not None:
            filtered_alerts = [a for a in filtered_alerts if a.resolved == resolved]
        
        return filtered_alerts[-limit:]
    
    async def _snapshot_loop(self) -> None:
        """Main snapshot collection loop"""
        while self.monitoring_active:
            try:
                await self.take_snapshot()
                await asyncio.sleep(self.config.snapshot_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Snapshot loop error", error=str(e))
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _alert_monitoring_loop(self) -> None:
        """Main alert monitoring loop"""
        while self.monitoring_active:
            try:
                await self._check_risk_alerts()
                await self._check_margin_alerts()
                await self._check_day_trading_alerts()
                await self._check_performance_alerts()
                
                await asyncio.sleep(self.config.alert_check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Alert monitoring loop error", error=str(e))
                await asyncio.sleep(5)
    
    async def _performance_calculation_loop(self) -> None:
        """Performance calculation loop"""
        while self.monitoring_active:
            try:
                self._update_performance_metrics()
                await asyncio.sleep(self.config.performance_calculation_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Performance calculation loop error", error=str(e))
                await asyncio.sleep(30)  # Longer pause for performance calc errors
    
    async def _check_risk_alerts(self) -> None:
        """Check for risk-related alerts"""
        if not self.current_snapshot:
            return
        
        snapshot = self.current_snapshot
        
        # Check buying power
        if snapshot.buying_power < self.config.min_buying_power_threshold:
            await self.create_alert(
                "low_buying_power",
                AlertLevel.WARNING,
                f"Buying power is low: ${snapshot.buying_power:,.2f}",
                {"buying_power": snapshot.buying_power, "threshold": self.config.min_buying_power_threshold}
            )
        
        # Check margin utilization
        if snapshot.margin_utilization > self.config.max_margin_utilization * 100:
            await self.create_alert(
                "high_margin_utilization",
                AlertLevel.CRITICAL,
                f"Margin utilization is high: {snapshot.margin_utilization:.1f}%",
                {"margin_utilization": snapshot.margin_utilization}
            )
        
        # Check for negative net liquidity
        if snapshot.net_liquidity < 0:
            await self.create_alert(
                "negative_net_liquidity",
                AlertLevel.EMERGENCY,
                f"Negative net liquidity: ${snapshot.net_liquidity:,.2f}",
                {"net_liquidity": snapshot.net_liquidity}
            )
    
    async def _check_margin_alerts(self) -> None:
        """Check for margin-related alerts"""
        if not self.current_snapshot:
            return
        
        snapshot = self.current_snapshot
        
        # Check maintenance margin breach
        if snapshot.equity < snapshot.maintenance_margin:
            await self.create_alert(
                "maintenance_margin_call",
                AlertLevel.EMERGENCY,
                "Maintenance margin call: Equity below required margin",
                {
                    "equity": snapshot.equity,
                    "maintenance_margin": snapshot.maintenance_margin,
                    "deficit": snapshot.maintenance_margin - snapshot.equity
                }
            )
    
    async def _check_day_trading_alerts(self) -> None:
        """Check for day trading rule alerts"""
        if not self.current_snapshot:
            return
        
        snapshot = self.current_snapshot
        
        # Check day trade count
        if snapshot.day_trade_count >= self.config.max_day_trades_per_period:
            if snapshot.equity < self.config.pdt_equity_minimum:
                await self.create_alert(
                    "pattern_day_trader_violation",
                    AlertLevel.CRITICAL,
                    f"PDT violation risk: {snapshot.day_trade_count} day trades with equity ${snapshot.equity:,.2f}",
                    {
                        "day_trade_count": snapshot.day_trade_count,
                        "equity": snapshot.equity,
                        "pdt_minimum": self.config.pdt_equity_minimum
                    }
                )
    
    async def _check_performance_alerts(self) -> None:
        """Check for performance-related alerts"""
        if len(self.snapshot_history) < 2:
            return
        
        # Calculate daily P&L
        current = self.current_snapshot
        previous = None
        
        # Find snapshot from approximately 24 hours ago
        target_time = current.timestamp - timedelta(hours=24)
        for snapshot in reversed(self.snapshot_history):
            if snapshot.timestamp <= target_time:
                previous = snapshot
                break
        
        if not previous:
            return
        
        daily_pnl = current.equity - previous.equity
        daily_pnl_pct = (daily_pnl / previous.equity) * 100 if previous.equity > 0 else 0
        
        # Check daily loss threshold
        if daily_pnl_pct < -self.config.max_daily_loss_threshold * 100:
            await self.create_alert(
                "daily_loss_threshold",
                AlertLevel.WARNING,
                f"Daily loss threshold exceeded: {daily_pnl_pct:.2f}%",
                {
                    "daily_pnl": daily_pnl,
                    "daily_pnl_percent": daily_pnl_pct,
                    "threshold_percent": self.config.max_daily_loss_threshold * 100
                }
            )
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate current performance metrics"""
        if len(self.snapshot_history) < 2:
            return {}
        
        # Basic calculations
        current = self.current_snapshot
        start_snapshot = self.snapshot_history[0]
        
        if not current or not start_snapshot:
            return {}
        
        # Calculate returns
        total_return = (current.equity - start_snapshot.equity) / start_snapshot.equity * 100 if start_snapshot.equity > 0 else 0
        
        # Calculate volatility (if enough data)
        if len(self.snapshot_history) >= 10:
            equities = [s.equity for s in self.snapshot_history]
            returns = [
                (equities[i] - equities[i-1]) / equities[i-1] * 100
                for i in range(1, len(equities))
                if equities[i-1] > 0
            ]
            
            volatility = np.std(returns) if returns else 0
            sharpe_ratio = np.mean(returns) / volatility if volatility > 0 and returns else 0
        else:
            volatility = 0
            sharpe_ratio = 0
        
        # Calculate max drawdown
        peak_equity = start_snapshot.equity
        max_drawdown = 0.0
        for snapshot in self.snapshot_history:
            if snapshot.equity > peak_equity:
                peak_equity = snapshot.equity
            drawdown = (peak_equity - snapshot.equity) / peak_equity * 100 if peak_equity > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'total_return_percent': total_return,
            'volatility_percent': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_percent': max_drawdown,
            'current_equity': current.equity,
            'peak_equity': peak_equity
        }
    
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate current risk metrics"""
        if not self.current_snapshot:
            return {}
        
        snapshot = self.current_snapshot
        
        # Position concentration (if position tracker available)
        max_position_concentration = 0.0
        if self.position_tracker:
            portfolio_summary = self.position_tracker.get_portfolio_summary()
            concentration_data = portfolio_summary.get('portfolio_concentration', {})
            max_position_concentration = concentration_data.get('max_symbol_concentration', 0.0) * 100
        
        return {
            'margin_utilization_percent': snapshot.margin_utilization,
            'net_liquidity': snapshot.net_liquidity,
            'buying_power_ratio': snapshot.buying_power / snapshot.equity * 100 if snapshot.equity > 0 else 0,
            'cash_ratio': snapshot.cash / snapshot.equity * 100 if snapshot.equity > 0 else 0,
            'max_position_concentration_percent': max_position_concentration,
            'day_trade_count': snapshot.day_trade_count
        }
    
    def _update_performance_metrics(self) -> None:
        """Update stored performance metrics"""
        if not self.current_snapshot:
            return
        
        # Calculate daily P&L if we have enough history
        if len(self.snapshot_history) >= 2:
            current = self.current_snapshot
            previous_day = None
            
            # Find snapshot from ~24 hours ago
            target_time = current.timestamp - timedelta(hours=24)
            for snapshot in reversed(self.snapshot_history):
                if snapshot.timestamp <= target_time:
                    previous_day = snapshot
                    break
            
            if previous_day:
                daily_pnl = current.equity - previous_day.equity
                daily_pnl_pct = (daily_pnl / previous_day.equity) * 100 if previous_day.equity > 0 else 0
                
                # Store daily P&L
                self.daily_pnl_history.append({
                    'date': current.timestamp.date(),
                    'pnl': daily_pnl,
                    'pnl_percent': daily_pnl_pct
                })
        
        # Clean up old data
        self._cleanup_old_data()
    
    def _cleanup_old_data(self) -> None:
        """Clean up old monitoring data"""
        cutoff_date = datetime.now() - timedelta(days=max(
            self.config.snapshot_retention_days,
            self.config.alert_retention_days
        ))
        
        # Clean up old alerts
        self.alert_history.extend([
            alert for alert in self.alerts
            if alert.timestamp < cutoff_date
        ])
        
        self.alerts = [
            alert for alert in self.alerts
            if alert.timestamp >= cutoff_date
        ]
        
        logger.debug("Cleaned up old monitoring data",
                    alerts_archived=len(self.alert_history),
                    active_alerts=len(self.alerts))