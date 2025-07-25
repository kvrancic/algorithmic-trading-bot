"""
Configuration Management

Handles loading and validation of configuration from YAML files
with environment variable substitution.
"""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)


class ConfigSection:
    """Base class for configuration sections"""
    
    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                # Convert nested dicts to ConfigSection objects
                setattr(self, key, ConfigSection(value))
            else:
                setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default"""
        return getattr(self, key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigSection):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path or "config/config.yaml"
        self._config_data = {}
        
        # Load configuration
        self._load_config()
        
        # Create convenience attributes
        self._create_sections()
        
        logger.info("Configuration loaded", 
                   config_path=self.config_path,
                   sections=list(self._config_data.keys()))
    
    def _load_config(self) -> None:
        """Load configuration from YAML file"""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(config_file, 'r') as f:
                config_text = f.read()
            
            # Perform environment variable substitution
            config_text = self._substitute_env_vars(config_text)
            
            # Parse YAML
            self._config_data = yaml.safe_load(config_text)
            
            if not isinstance(self._config_data, dict):
                raise ValueError("Configuration file must contain a dictionary")
                
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")
    
    def _substitute_env_vars(self, text: str) -> str:
        """Substitute environment variables in configuration text"""
        import re
        
        # Pattern to match ${VAR_NAME} or ${VAR_NAME:default_value}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replace_env_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            
            return os.getenv(var_name, default_value)
        
        return re.sub(pattern, replace_env_var, text)
    
    def _create_sections(self) -> None:
        """Create configuration sections as attributes"""
        for section_name, section_data in self._config_data.items():
            if isinstance(section_data, dict):
                # Store sections in private dict to avoid property conflicts
                if not hasattr(self, '_sections'):
                    self._sections = {}
                self._sections[section_name] = ConfigSection(section_data)
            else:
                if not hasattr(self, '_sections'):
                    self._sections = {}
                self._sections[section_name] = section_data
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get top-level configuration value"""
        return self._config_data.get(key, default)
    
    def get_nested(self, path: str, default: Any = None) -> Any:
        """
        Get nested configuration value using dot notation
        
        Example: config.get_nested('trading.max_position_size')
        """
        keys = path.split('.')
        value = self._config_data
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def update(self, section: str, updates: Dict[str, Any]) -> None:
        """Update configuration section"""
        if section not in self._config_data:
            self._config_data[section] = {}
        
        self._config_data[section].update(updates)
        
        # Recreate sections
        self._create_sections()
        
        logger.info("Configuration updated", section=section, updates=updates)
    
    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary"""
        return self._config_data.copy()
    
    def validate(self) -> bool:
        """Validate configuration"""
        required_sections = ['trading', 'risk', 'database']
        
        for section in required_sections:
            if section not in self._config_data:
                logger.error("Missing required configuration section", section=section)
                return False
        
        # Validate trading section
        if not self._validate_trading_config():
            return False
        
        # Validate risk section
        if not self._validate_risk_config():
            return False
        
        return True
    
    def _validate_trading_config(self) -> bool:
        """Validate trading configuration"""
        trading = self._config_data.get('trading', {})
        
        # Check for required fields
        required_fields = ['watchlist', 'max_positions', 'max_position_size']
        for field in required_fields:
            if field not in trading:
                logger.error("Missing required trading config", field=field)
                return False
        
        # Validate watchlist
        watchlist = trading.get('watchlist', [])
        if not isinstance(watchlist, list) or not watchlist:
            logger.error("Watchlist must be a non-empty list")
            return False
        
        # Validate numeric fields
        if trading.get('max_positions', 0) <= 0:
            logger.error("max_positions must be positive")
            return False
        
        if not (0 < trading.get('max_position_size', 0) <= 1):
            logger.error("max_position_size must be between 0 and 1")
            return False
        
        return True
    
    def _validate_risk_config(self) -> bool:
        """Validate risk configuration"""
        risk = self._config_data.get('risk', {})
        
        # Check for required fields
        required_fields = ['max_drawdown', 'risk_per_trade', 'stop_loss_pct']
        for field in required_fields:
            if field not in risk:
                logger.error("Missing required risk config", field=field)
                return False
        
        # Validate numeric ranges
        if not (0 < risk.get('max_drawdown', 0) <= 1):
            logger.error("max_drawdown must be between 0 and 1")
            return False
        
        if not (0 < risk.get('risk_per_trade', 0) <= 0.1):
            logger.error("risk_per_trade must be between 0 and 0.1")
            return False
        
        if not (0 < risk.get('stop_loss_pct', 0) <= 0.2):
            logger.error("stop_loss_pct must be between 0 and 0.2")
            return False
        
        return True
    
    @property 
    def trading(self) -> ConfigSection:
        """Trading configuration section"""
        return self._sections.get('trading', ConfigSection({}))
    
    @property
    def risk(self) -> ConfigSection:
        """Risk management configuration section"""
        return self._sections.get('risk', ConfigSection({}))
    
    @property
    def database(self) -> ConfigSection:
        """Database configuration section"""
        return self._sections.get('database', ConfigSection({}))
    
    @property
    def data_sources(self) -> ConfigSection:
        """Data sources configuration section"""
        return self._sections.get('data_sources', ConfigSection({}))
    
    @property
    def ml(self) -> ConfigSection:
        """Machine learning configuration section"""
        return self._sections.get('ml', ConfigSection({}))
    
    @property
    def execution(self) -> ConfigSection:
        """Execution configuration section"""
        return self._sections.get('execution', ConfigSection({}))
    
    @property
    def paths(self) -> ConfigSection:
        """Paths configuration section"""
        return self._sections.get('paths', ConfigSection({
            'data': 'data/',
            'models': 'models/',
            'logs': 'logs/',
            'reports': 'reports/'
        }))
    
    @property
    def universe(self) -> ConfigSection:
        """Universe configuration section"""
        return self._sections.get('universe', ConfigSection({}))


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded configuration object
    """
    config = Config(config_path)
    
    if not config.validate():
        raise ValueError("Configuration validation failed")
    
    return config


# Default configuration instance
_default_config: Optional[Config] = None


def get_config() -> Config:
    """Get the default configuration instance"""
    global _default_config
    
    if _default_config is None:
        _default_config = load_config()
    
    return _default_config


def set_config(config: Config) -> None:
    """Set the default configuration instance"""
    global _default_config
    _default_config = config