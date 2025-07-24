"""
Base strategy class for trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params or {}
        self.positions = {}
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from market data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals (1 for buy, -1 for sell, 0 for hold)
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: float, price: float, 
                              portfolio_value: float) -> float:
        """
        Calculate the position size for a given signal.
        
        Args:
            signal: Trading signal (-1, 0, 1)
            price: Current price
            portfolio_value: Total portfolio value
            
        Returns:
            Position size (positive for long, negative for short)
        """
        pass
    
    def update_positions(self, symbol: str, position_size: float):
        """Update the current positions."""
        self.positions[symbol] = position_size
    
    def get_positions(self) -> Dict[str, float]:
        """Get current positions."""
        return self.positions.copy() 