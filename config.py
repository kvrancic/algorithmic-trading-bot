"""
Configuration settings for the algorithmic trading bot.
"""

import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration class for the trading bot."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        self.logs_dir = self.base_dir / "logs"
        
        # Create necessary directories
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Trading parameters
        self.symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        self.timeframe = "1d"
        self.lookback_period = 30
        
        # Risk management
        self.max_position_size = 0.1  # 10% of portfolio
        self.stop_loss_pct = 0.05     # 5% stop loss
        self.take_profit_pct = 0.15   # 15% take profit
        
        # API settings (load from environment variables)
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        self.secret_key = os.getenv("TRADING_SECRET_KEY", "")
        
    def get_trading_params(self) -> Dict[str, Any]:
        """Get trading parameters as a dictionary."""
        return {
            "symbols": self.symbols,
            "timeframe": self.timeframe,
            "lookback_period": self.lookback_period,
            "max_position_size": self.max_position_size,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
        }


# Global configuration instance
config = Config() 