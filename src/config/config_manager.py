"""
Configuration Manager Adapter

Provides a standardized interface for configuration access across different components.
Acts as an adapter for the existing Config class to work with new components.
"""

from typing import Any, Dict
from ..config import Config


class ConfigManager:
    """Configuration manager adapter for existing Config class"""
    
    def __init__(self, config: Config):
        """
        Initialize with existing Config instance
        
        Args:
            config: Existing Config instance
        """
        self._config = config
    
    @property
    def config(self) -> Config:
        """Get the underlying config object"""
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key
        
        Args:
            key: Configuration key in dot notation (e.g., 'trading.strategy_mode')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            # Split the key and traverse the config object
            keys = key.split('.')
            value = self._config
            
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                elif isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception:
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """
        Set configuration value by dot-notation key
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            keys = key.split('.')
            if len(keys) == 1:
                setattr(self._config, keys[0], value)
                return True
            
            # Navigate to parent object
            obj = self._config
            for k in keys[:-1]:
                if hasattr(obj, k):
                    obj = getattr(obj, k)
                elif isinstance(obj, dict) and k in obj:
                    obj = obj[k]
                else:
                    return False
            
            # Set the final value
            final_key = keys[-1]
            if hasattr(obj, final_key):
                setattr(obj, final_key, value)
                return True
            elif isinstance(obj, dict):
                obj[final_key] = value
                return True
            
            return False
            
        except Exception:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format
        
        Returns:
            Configuration as dictionary
        """
        try:
            # Try to convert config to dict if it has a to_dict method
            if hasattr(self._config, 'to_dict'):
                return self._config.to_dict()
            
            # Otherwise, use object attributes
            result = {}
            for attr in dir(self._config):
                if not attr.startswith('_') and not callable(getattr(self._config, attr)):
                    value = getattr(self._config, attr)
                    result[attr] = value
            
            return result
            
        except Exception:
            return {}