"""
Model Performance Monitoring System

Real-time monitoring and alerting system for model performance.
Tracks drift, degradation, and anomalies in model behavior.
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from collections import deque, defaultdict
import threading
import time
import json
import structlog
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import warnings

from ..models.base import BaseModel

logger = structlog.get_logger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for performance monitoring"""
    
    # Monitoring intervals
    check_interval_minutes: int = 15  # How often to check performance
    evaluation_window_hours: int = 24  # Window for performance evaluation
    baseline_window_days: int = 30  # Window for establishing baseline
    
    # Thresholds for alerts
    performance_degradation_threshold: float = 0.10  # 10% degradation
    accuracy_threshold: float = 0.50  # Minimum acceptable accuracy
    sharpe_threshold: float = 0.5  # Minimum Sharpe ratio
    drawdown_threshold: float = 0.20  # Maximum acceptable drawdown
    
    # Drift detection
    enable_drift_detection: bool = True
    drift_detection_method: str = "ks_test"  # "ks_test", "psi", "js_divergence"
    drift_threshold: float = 0.05  # P-value threshold for statistical tests
    
    # Alert settings
    enable_alerts: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["log", "email"])
    email_recipients: List[str] = field(default_factory=list)
    webhook_url: Optional[str] = None
    
    # Data storage
    metrics_storage_path: Path = field(default_factory=lambda: Path("monitoring_data"))
    max_stored_days: int = 90  # How long to keep monitoring data
    
    # Model comparison
    enable_model_comparison: bool = True
    comparison_metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'sharpe_ratio', 'max_drawdown', 'hit_rate'
    ])
    
    def __post_init__(self):
        self.metrics_storage_path = Path(self.metrics_storage_path)
        self.metrics_storage_path.mkdir(exist_ok=True, parents=True)


class PerformanceMetric:
    """Individual performance metric tracking"""
    
    def __init__(self, name: str, value: float, timestamp: datetime, metadata: Dict[str, Any] = None):
        self.name = name
        self.value = value
        self.timestamp = timestamp
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetric':
        return cls(
            name=data['name'],
            value=data['value'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )


class Alert:
    """Performance alert"""
    
    def __init__(
        self,
        alert_type: str,
        severity: str,
        message: str,
        model_name: str,
        metric_name: str,
        current_value: float,
        threshold_value: float,
        timestamp: datetime = None
    ):
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.model_name = model_name
        self.metric_name = metric_name
        self.current_value = current_value
        self.threshold_value = threshold_value
        self.timestamp = timestamp or datetime.now()
        self.alert_id = self._generate_alert_id()
    
    def _generate_alert_id(self) -> str:
        import hashlib
        content = f"{self.alert_type}_{self.model_name}_{self.metric_name}_{self.timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'model_name': self.model_name,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'timestamp': self.timestamp.isoformat()
        }


class DriftDetector:
    """Statistical drift detection for model inputs and outputs"""
    
    def __init__(self, method: str = "ks_test"):
        self.method = method
        self.baseline_data: Dict[str, np.ndarray] = {}
    
    def set_baseline(self, data: Dict[str, np.ndarray]):
        """Set baseline data for drift comparison"""
        self.baseline_data = {key: np.array(values) for key, values in data.items()}
        logger.info("Drift detection baseline set", features=list(data.keys()))
    
    def detect_drift(self, new_data: Dict[str, np.ndarray], threshold: float = 0.05) -> Dict[str, Dict[str, Any]]:
        """Detect drift in new data compared to baseline"""
        
        drift_results = {}
        
        for feature_name, new_values in new_data.items():
            if feature_name not in self.baseline_data:
                continue
            
            baseline_values = self.baseline_data[feature_name]
            drift_result = self._calculate_drift(baseline_values, new_values, threshold)
            drift_results[feature_name] = drift_result
        
        return drift_results
    
    def _calculate_drift(self, baseline: np.ndarray, new_data: np.ndarray, threshold: float) -> Dict[str, Any]:
        """Calculate drift using specified method"""
        
        if self.method == "ks_test":
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(baseline, new_data)
            drift_detected = p_value < threshold
            
            return {
                'method': 'ks_test',
                'statistic': statistic,
                'p_value': p_value,
                'drift_detected': drift_detected,
                'severity': self._get_drift_severity(p_value, threshold)
            }
        
        elif self.method == "psi":
            # Population Stability Index
            psi_value = self._calculate_psi(baseline, new_data)
            drift_detected = psi_value > 0.2  # Common PSI threshold
            
            return {
                'method': 'psi',
                'psi_value': psi_value,
                'drift_detected': drift_detected,
                'severity': self._get_psi_severity(psi_value)
            }
        
        elif self.method == "js_divergence":
            # Jensen-Shannon divergence
            js_div = self._calculate_js_divergence(baseline, new_data)
            drift_detected = js_div > 0.1  # Threshold for JS divergence
            
            return {
                'method': 'js_divergence',
                'js_divergence': js_div,
                'drift_detected': drift_detected,
                'severity': self._get_js_severity(js_div)
            }
        
        else:
            raise ValueError(f"Unknown drift detection method: {self.method}")
    
    def _calculate_psi(self, baseline: np.ndarray, new_data: np.ndarray, buckets: int = 10) -> float:
        """Calculate Population Stability Index"""
        
        # Create bins based on baseline data
        _, bin_edges = np.histogram(baseline, bins=buckets)
        
        # Calculate distributions
        baseline_dist, _ = np.histogram(baseline, bins=bin_edges)
        new_dist, _ = np.histogram(new_data, bins=bin_edges)
        
        # Normalize to get proportions
        baseline_prop = baseline_dist / len(baseline)
        new_prop = new_dist / len(new_data)
        
        # Avoid division by zero
        baseline_prop = np.where(baseline_prop == 0, 0.0001, baseline_prop)
        new_prop = np.where(new_prop == 0, 0.0001, new_prop)
        
        # Calculate PSI
        psi = np.sum((new_prop - baseline_prop) * np.log(new_prop / baseline_prop))
        
        return psi
    
    def _calculate_js_divergence(self, baseline: np.ndarray, new_data: np.ndarray, bins: int = 50) -> float:
        """Calculate Jensen-Shannon divergence"""
        
        # Create normalized histograms
        range_min = min(baseline.min(), new_data.min())
        range_max = max(baseline.max(), new_data.max())
        
        hist_baseline, _ = np.histogram(baseline, bins=bins, range=(range_min, range_max), density=True)
        hist_new, _ = np.histogram(new_data, bins=bins, range=(range_min, range_max), density=True)
        
        # Normalize to probabilities
        hist_baseline = hist_baseline / hist_baseline.sum()
        hist_new = hist_new / hist_new.sum()
        
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        hist_baseline = hist_baseline + eps
        hist_new = hist_new + eps
        
        # Calculate JS divergence
        m = 0.5 * (hist_baseline + hist_new)
        js_div = 0.5 * stats.entropy(hist_baseline, m) + 0.5 * stats.entropy(hist_new, m)
        
        return js_div
    
    def _get_drift_severity(self, p_value: float, threshold: float) -> str:
        """Determine drift severity based on p-value"""
        if p_value < threshold / 10:
            return "high"
        elif p_value < threshold / 2:
            return "medium"
        elif p_value < threshold:
            return "low"
        else:
            return "none"
    
    def _get_psi_severity(self, psi_value: float) -> str:
        """Determine drift severity based on PSI value"""
        if psi_value > 0.25:
            return "high"
        elif psi_value > 0.2:
            return "medium"
        elif psi_value > 0.1:
            return "low"
        else:
            return "none"
    
    def _get_js_severity(self, js_value: float) -> str:
        """Determine drift severity based on JS divergence"""
        if js_value > 0.2:
            return "high"
        elif js_value > 0.1:
            return "medium"
        elif js_value > 0.05:
            return "low"
        else:
            return "none"


class ModelMonitor:
    """Monitor for individual model performance"""
    
    def __init__(self, model_name: str, model: BaseModel, config: MonitoringConfig):
        self.model_name = model_name
        self.model = model
        self.config = config
        
        # Performance tracking
        self.metrics_history: deque = deque(maxlen=10000)  # Keep last 10k metrics
        self.baseline_metrics: Dict[str, float] = {}
        self.current_metrics: Dict[str, float] = {}
        
        # Drift detection
        self.drift_detector = DriftDetector(config.drift_detection_method)
        self.drift_baseline_set = False
        
        # Alert tracking
        self.recent_alerts: deque = deque(maxlen=100)
        self.alert_cooldowns: Dict[str, datetime] = {}
        
        # Performance data buffers
        self.prediction_buffer: deque = deque(maxlen=1000)
        self.feature_buffer: deque = deque(maxlen=1000)
        
        logger.info(f"Model monitor initialized for {model_name}")
    
    def record_prediction(
        self,
        features: np.ndarray,
        prediction: np.ndarray,
        actual: Optional[np.ndarray] = None,
        timestamp: datetime = None
    ):
        """Record a prediction for monitoring"""
        
        timestamp = timestamp or datetime.now()
        
        # Store in buffers
        self.prediction_buffer.append({
            'features': features,
            'prediction': prediction,
            'actual': actual,
            'timestamp': timestamp
        })
        
        # Store features for drift detection
        if len(features.shape) == 1:
            self.feature_buffer.append(features)
        else:
            # For batch predictions, take the mean
            self.feature_buffer.append(features.mean(axis=0))
    
    def calculate_current_metrics(self) -> Dict[str, float]:
        """Calculate current performance metrics"""
        
        if len(self.prediction_buffer) == 0:
            return {}
        
        # Get recent predictions with actuals
        recent_with_actuals = [
            entry for entry in self.prediction_buffer 
            if entry['actual'] is not None
        ]
        
        if len(recent_with_actuals) == 0:
            return {}
        
        # Extract predictions and actuals
        predictions = np.array([entry['prediction'] for entry in recent_with_actuals])
        actuals = np.array([entry['actual'] for entry in recent_with_actuals])
        
        # Flatten if needed
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        if actuals.ndim > 1:
            actuals = actuals.flatten()
        
        metrics = {}
        
        # Accuracy (for classification)
        if len(np.unique(actuals)) <= 20:  # Assume classification
            metrics['accuracy'] = np.mean(predictions == actuals)
            
            # Precision, recall, F1 (binary classification)
            if len(np.unique(actuals)) == 2:
                tp = np.sum((predictions == 1) & (actuals == 1))
                fp = np.sum((predictions == 1) & (actuals == 0))
                fn = np.sum((predictions == 0) & (actuals == 1))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics['precision'] = precision
                metrics['recall'] = recall
                metrics['f1_score'] = f1
        
        # Regression metrics
        else:
            mse = np.mean((predictions - actuals) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - actuals))
            
            metrics['mse'] = mse
            metrics['rmse'] = rmse
            metrics['mae'] = mae
            
            # R-squared
            ss_res = np.sum((actuals - predictions) ** 2)
            ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            metrics['r2_score'] = r2
        
        # Trading-specific metrics (if predictions represent returns)
        if len(predictions) > 1:
            # Treat predictions as returns
            returns = pd.Series(predictions)
            
            # Sharpe ratio
            if returns.std() > 0:
                metrics['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252)
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            metrics['max_drawdown'] = drawdown.min()
            
            # Hit rate
            positive_returns = returns > 0
            metrics['hit_rate'] = positive_returns.mean()
            
            # Volatility
            metrics['volatility'] = returns.std() * np.sqrt(252)
        
        self.current_metrics = metrics
        
        # Record metrics in history
        for metric_name, value in metrics.items():
            metric = PerformanceMetric(
                name=metric_name,
                value=value,
                timestamp=datetime.now(),
                metadata={'model_name': self.model_name}
            )
            self.metrics_history.append(metric)
        
        return metrics
    
    def set_baseline_metrics(self, metrics: Dict[str, float]):
        """Set baseline metrics for comparison"""
        self.baseline_metrics = metrics.copy()
        logger.info(f"Baseline metrics set for {self.model_name}", metrics=metrics)
    
    def check_performance_degradation(self) -> List[Alert]:
        """Check for performance degradation and generate alerts"""
        
        alerts = []
        
        if not self.baseline_metrics or not self.current_metrics:
            return alerts
        
        # Check each metric for degradation
        for metric_name in self.current_metrics:
            if metric_name not in self.baseline_metrics:
                continue
            
            current_value = self.current_metrics[metric_name]
            baseline_value = self.baseline_metrics[metric_name]
            
            # Calculate degradation (handle different metric directions)
            if metric_name in ['max_drawdown', 'mse', 'rmse', 'mae', 'volatility']:
                # Lower is better - degradation is increase
                degradation = (current_value - baseline_value) / abs(baseline_value) if baseline_value != 0 else 0
            else:
                # Higher is better - degradation is decrease
                degradation = (baseline_value - current_value) / abs(baseline_value) if baseline_value != 0 else 0
            
            # Check if degradation exceeds threshold
            if degradation > self.config.performance_degradation_threshold:
                # Check cooldown to avoid spam
                cooldown_key = f"{metric_name}_degradation"
                if self._check_alert_cooldown(cooldown_key):
                    
                    severity = "high" if degradation > 0.25 else "medium" if degradation > 0.15 else "low"
                    
                    alert = Alert(
                        alert_type="performance_degradation",
                        severity=severity,
                        message=f"{metric_name} degraded by {degradation:.2%} from baseline",
                        model_name=self.model_name,
                        metric_name=metric_name,
                        current_value=current_value,
                        threshold_value=baseline_value
                    )
                    
                    alerts.append(alert)
                    self._set_alert_cooldown(cooldown_key)
        
        return alerts
    
    def check_threshold_violations(self) -> List[Alert]:
        """Check for threshold violations"""
        
        alerts = []
        
        threshold_checks = [
            ('accuracy', self.config.accuracy_threshold, 'minimum', 'high'),
            ('sharpe_ratio', self.config.sharpe_threshold, 'minimum', 'medium'),
            ('max_drawdown', -self.config.drawdown_threshold, 'maximum', 'high'),
        ]
        
        for metric_name, threshold, check_type, severity in threshold_checks:
            if metric_name not in self.current_metrics:
                continue
            
            current_value = self.current_metrics[metric_name]
            violation = False
            
            if check_type == 'minimum' and current_value < threshold:
                violation = True
                message = f"{metric_name} ({current_value:.3f}) below minimum threshold ({threshold:.3f})"
            elif check_type == 'maximum' and current_value > threshold:
                violation = True
                message = f"{metric_name} ({current_value:.3f}) above maximum threshold ({threshold:.3f})"
            
            if violation:
                cooldown_key = f"{metric_name}_threshold"
                if self._check_alert_cooldown(cooldown_key):
                    alert = Alert(
                        alert_type="threshold_violation",
                        severity=severity,
                        message=message,
                        model_name=self.model_name,
                        metric_name=metric_name,
                        current_value=current_value,
                        threshold_value=threshold
                    )
                    
                    alerts.append(alert)
                    self._set_alert_cooldown(cooldown_key)
        
        return alerts
    
    def check_drift(self) -> List[Alert]:
        """Check for feature drift"""
        
        alerts = []
        
        if not self.config.enable_drift_detection or len(self.feature_buffer) < 100:
            return alerts
        
        # Set baseline if not set
        if not self.drift_baseline_set and len(self.feature_buffer) >= 500:
            baseline_features = np.array(list(self.feature_buffer)[:500])
            if baseline_features.ndim == 2:
                baseline_data = {f'feature_{i}': baseline_features[:, i] 
                               for i in range(baseline_features.shape[1])}
            else:
                baseline_data = {'feature_0': baseline_features}
            
            self.drift_detector.set_baseline(baseline_data)
            self.drift_baseline_set = True
            return alerts
        
        if not self.drift_baseline_set:
            return alerts
        
        # Get recent features for drift check
        recent_features = np.array(list(self.feature_buffer)[-100:])
        if recent_features.ndim == 2:
            new_data = {f'feature_{i}': recent_features[:, i] 
                       for i in range(recent_features.shape[1])}
        else:
            new_data = {'feature_0': recent_features}
        
        # Detect drift
        drift_results = self.drift_detector.detect_drift(new_data, self.config.drift_threshold)
        
        # Generate alerts for detected drift
        for feature_name, drift_result in drift_results.items():
            if drift_result['drift_detected']:
                cooldown_key = f"drift_{feature_name}"
                if self._check_alert_cooldown(cooldown_key):
                    
                    if drift_result['method'] == 'ks_test':
                        detail = f"p-value: {drift_result['p_value']:.4f}"
                    elif drift_result['method'] == 'psi':
                        detail = f"PSI: {drift_result['psi_value']:.4f}"
                    else:
                        detail = f"JS divergence: {drift_result['js_divergence']:.4f}"
                    
                    alert = Alert(
                        alert_type="drift_detected",
                        severity=drift_result['severity'],
                        message=f"Drift detected in {feature_name} ({detail})",
                        model_name=self.model_name,
                        metric_name=feature_name,
                        current_value=0,  # Not applicable for drift
                        threshold_value=self.config.drift_threshold
                    )
                    
                    alerts.append(alert)
                    self._set_alert_cooldown(cooldown_key)
        
        return alerts
    
    def _check_alert_cooldown(self, alert_key: str, cooldown_minutes: int = 60) -> bool:
        """Check if alert is in cooldown period"""
        
        if alert_key not in self.alert_cooldowns:
            return True
        
        cooldown_end = self.alert_cooldowns[alert_key] + timedelta(minutes=cooldown_minutes)
        return datetime.now() > cooldown_end
    
    def _set_alert_cooldown(self, alert_key: str):
        """Set alert cooldown"""
        self.alert_cooldowns[alert_key] = datetime.now()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of model performance"""
        
        return {
            'model_name': self.model_name,
            'current_metrics': self.current_metrics,
            'baseline_metrics': self.baseline_metrics,
            'metrics_history_length': len(self.metrics_history),
            'recent_predictions': len(self.prediction_buffer),
            'drift_baseline_set': self.drift_baseline_set,
            'recent_alerts': len(self.recent_alerts)
        }


class PerformanceMonitor:
    """Main performance monitoring system"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.model_monitors: Dict[str, ModelMonitor] = {}
        self.alerts_history: deque = deque(maxlen=1000)
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Alert handlers
        self.alert_handlers = {
            'log': self._log_alert,
            'email': self._send_email_alert,
            'webhook': self._send_webhook_alert
        }
    
    def add_model(self, model_name: str, model: BaseModel) -> ModelMonitor:
        """Add a model to monitoring"""
        
        monitor = ModelMonitor(model_name, model, self.config)
        self.model_monitors[model_name] = monitor
        
        logger.info(f"Added model to monitoring: {model_name}")
        return monitor
    
    def remove_model(self, model_name: str):
        """Remove a model from monitoring"""
        
        if model_name in self.model_monitors:
            del self.model_monitors[model_name]
            logger.info(f"Removed model from monitoring: {model_name}")
    
    def start_monitoring(self):
        """Start the monitoring loop"""
        
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring loop"""
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        
        while self.monitoring_active:
            try:
                self._check_all_models()
                time.sleep(self.config.check_interval_minutes * 60)
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                time.sleep(60)  # Wait a minute before retrying
    
    def _check_all_models(self):
        """Check all monitored models"""
        
        all_alerts = []
        
        for model_name, monitor in self.model_monitors.items():
            try:
                # Calculate current metrics
                monitor.calculate_current_metrics()
                
                # Check for issues
                alerts = []
                alerts.extend(monitor.check_performance_degradation())
                alerts.extend(monitor.check_threshold_violations())
                alerts.extend(monitor.check_drift())
                
                # Store alerts
                for alert in alerts:
                    monitor.recent_alerts.append(alert)
                    self.alerts_history.append(alert)
                
                all_alerts.extend(alerts)
                
            except Exception as e:
                logger.error(f"Error monitoring model {model_name}", error=str(e))
        
        # Process alerts
        if all_alerts:
            self._process_alerts(all_alerts)
    
    def _process_alerts(self, alerts: List[Alert]):
        """Process and send alerts"""
        
        if not self.config.enable_alerts:
            return
        
        # Group alerts by severity
        high_alerts = [a for a in alerts if a.severity == 'high']
        medium_alerts = [a for a in alerts if a.severity == 'medium']
        low_alerts = [a for a in alerts if a.severity == 'low']
        
        # Send alerts through configured channels
        for channel in self.config.alert_channels:
            if channel in self.alert_handlers:
                try:
                    self.alert_handlers[channel](alerts)
                except Exception as e:
                    logger.error(f"Failed to send alert via {channel}", error=str(e))
        
        logger.info("Alerts processed",
                   high=len(high_alerts),
                   medium=len(medium_alerts),
                   low=len(low_alerts))
    
    def _log_alert(self, alerts: List[Alert]):
        """Log alerts"""
        
        for alert in alerts:
            if alert.severity == 'high':
                logger.error("HIGH SEVERITY ALERT", **alert.to_dict())
            elif alert.severity == 'medium':
                logger.warning("MEDIUM SEVERITY ALERT", **alert.to_dict())
            else:
                logger.info("LOW SEVERITY ALERT", **alert.to_dict())
    
    def _send_email_alert(self, alerts: List[Alert]):
        """Send email alerts (placeholder implementation)"""
        
        if not self.config.email_recipients:
            return
        
        # This would integrate with an email service
        # For now, just log that email would be sent
        high_alerts = [a for a in alerts if a.severity == 'high']
        if high_alerts:
            logger.info("Would send email alert",
                       recipients=self.config.email_recipients,
                       alert_count=len(high_alerts))
    
    def _send_webhook_alert(self, alerts: List[Alert]):
        """Send webhook alerts (placeholder implementation)"""
        
        if not self.config.webhook_url:
            return
        
        # This would send HTTP POST to webhook URL
        # For now, just log that webhook would be called
        high_alerts = [a for a in alerts if a.severity == 'high']
        if high_alerts:
            logger.info("Would send webhook alert",
                       webhook_url=self.config.webhook_url,
                       alert_count=len(high_alerts))
    
    def record_prediction(
        self,
        model_name: str,
        features: np.ndarray,
        prediction: np.ndarray,
        actual: Optional[np.ndarray] = None,
        timestamp: datetime = None
    ):
        """Record a prediction for monitoring"""
        
        if model_name in self.model_monitors:
            self.model_monitors[model_name].record_prediction(
                features, prediction, actual, timestamp
            )
    
    def set_model_baseline(self, model_name: str, metrics: Dict[str, float]):
        """Set baseline metrics for a model"""
        
        if model_name in self.model_monitors:
            self.model_monitors[model_name].set_baseline_metrics(metrics)
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get monitoring dashboard data"""
        
        dashboard = {
            'monitoring_active': self.monitoring_active,
            'total_models': len(self.model_monitors),
            'total_alerts': len(self.alerts_history),
            'models': {}
        }
        
        # Get summary for each model
        for model_name, monitor in self.model_monitors.items():
            dashboard['models'][model_name] = monitor.get_performance_summary()
        
        # Recent alerts summary
        recent_alerts = [a for a in self.alerts_history if a.timestamp > datetime.now() - timedelta(hours=24)]
        dashboard['recent_alerts'] = {
            'total': len(recent_alerts),
            'high': len([a for a in recent_alerts if a.severity == 'high']),
            'medium': len([a for a in recent_alerts if a.severity == 'medium']),
            'low': len([a for a in recent_alerts if a.severity == 'low'])
        }
        
        return dashboard
    
    def export_monitoring_data(self, path: Path, days: int = 7):
        """Export monitoring data to file"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'export_period_days': days,
            'models': {},
            'alerts': []
        }
        
        # Export model data
        for model_name, monitor in self.model_monitors.items():
            recent_metrics = [
                metric.to_dict() for metric in monitor.metrics_history
                if metric.timestamp > cutoff_date
            ]
            
            export_data['models'][model_name] = {
                'current_metrics': monitor.current_metrics,
                'baseline_metrics': monitor.baseline_metrics,
                'metrics_history': recent_metrics,
                'drift_baseline_set': monitor.drift_baseline_set
            }
        
        # Export alerts
        recent_alerts = [
            alert.to_dict() for alert in self.alerts_history
            if alert.timestamp > cutoff_date
        ]
        export_data['alerts'] = recent_alerts
        
        # Save to file
        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info("Monitoring data exported", path=str(path), days=days)
    
    def cleanup_old_data(self, days: int = None):
        """Clean up old monitoring data"""
        
        cleanup_days = days or self.config.max_stored_days
        cutoff_date = datetime.now() - timedelta(days=cleanup_days)
        
        # Clean up alerts
        initial_alert_count = len(self.alerts_history)
        self.alerts_history = deque([
            alert for alert in self.alerts_history
            if alert.timestamp > cutoff_date
        ], maxlen=1000)
        
        alerts_removed = initial_alert_count - len(self.alerts_history)
        
        # Clean up model metrics
        total_metrics_removed = 0
        for monitor in self.model_monitors.values():
            initial_count = len(monitor.metrics_history)
            monitor.metrics_history = deque([
                metric for metric in monitor.metrics_history
                if metric.timestamp > cutoff_date
            ], maxlen=10000)
            total_metrics_removed += initial_count - len(monitor.metrics_history)
        
        logger.info("Cleaned up old monitoring data",
                   alerts_removed=alerts_removed,
                   metrics_removed=total_metrics_removed,
                   cutoff_days=cleanup_days)