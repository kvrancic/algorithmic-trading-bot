"""
Trading strategies package for the algorithmic trading bot.
"""

from .base_strategy import BaseStrategy
from .moving_average_strategy import MovingAverageStrategy

__all__ = ['BaseStrategy', 'MovingAverageStrategy'] 