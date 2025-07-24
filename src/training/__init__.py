"""
Training Module for QuantumSentiment Trading Bot

This module provides comprehensive training infrastructure including:
- ModelTrainingPipeline: Orchestrates training of all models
- WalkForwardOptimizer: Implements time-series cross-validation
- ModelValidator: Validates model performance
- ModelPersistence: Handles model saving/loading
- PerformanceMonitor: Tracks model performance over time
"""

from .training_pipeline import ModelTrainingPipeline, TrainingConfig
from .walk_forward_optimizer import WalkForwardOptimizer, WalkForwardConfig
from .model_validator import ModelValidator, ValidationConfig
from .model_persistence import ModelPersistence, PersistenceConfig
from .performance_monitor import PerformanceMonitor, MonitoringConfig

__all__ = [
    'ModelTrainingPipeline', 'TrainingConfig',
    'WalkForwardOptimizer', 'WalkForwardConfig', 
    'ModelValidator', 'ValidationConfig',
    'ModelPersistence', 'PersistenceConfig',
    'PerformanceMonitor', 'MonitoringConfig'
]