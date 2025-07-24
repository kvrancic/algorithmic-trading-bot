"""
Moving Average Crossover Strategy
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_strategy import BaseStrategy


class MovingAverageStrategy(BaseStrategy):
    """Simple moving average crossover strategy."""
    
    def __init__(self, short_window: int = 10, long_window: int = 30, 
                 max_position_size: float = 0.1):
        params = {
            'short_window': short_window,
            'long_window': long_window,
            'max_position_size': max_position_size
        }
        super().__init__("Moving Average Crossover", params)
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on moving average crossover.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        df = data.copy()
        
        # Calculate moving averages
        short_ma = df['close'].rolling(window=self.params['short_window']).mean()
        long_ma = df['close'].rolling(window=self.params['long_window']).mean()
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        signals[short_ma > long_ma] = 1    # Buy signal
        signals[short_ma < long_ma] = -1   # Sell signal
        
        df['signal'] = signals
        df['short_ma'] = short_ma
        df['long_ma'] = long_ma
        
        return df
    
    def calculate_position_size(self, signal: float, price: float, 
                              portfolio_value: float) -> float:
        """
        Calculate position size based on signal and risk management.
        
        Args:
            signal: Trading signal (-1, 0, 1)
            price: Current price
            portfolio_value: Total portfolio value
            
        Returns:
            Position size
        """
        if signal == 0:
            return 0
            
        # Calculate position size as percentage of portfolio
        position_value = portfolio_value * self.params['max_position_size']
        position_size = position_value / price
        
        # Apply signal direction
        return position_size * signal 