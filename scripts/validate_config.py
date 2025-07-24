#!/usr/bin/env python3
"""
Configuration Validation Script for QuantumSentiment Trading Bot

This script validates the configuration files and environment variables
to ensure all required settings are present and properly configured.

Usage:
    python scripts/validate_config.py
    python scripts/validate_config.py --config config/custom.yaml
    python scripts/validate_config.py --env-file custom.env
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, validator
import requests


class TradingConfig(BaseModel):
    """Trading configuration validation"""
    mode: str
    initial_capital: float
    max_position_size: float
    max_daily_loss: float
    max_portfolio_risk: float
    max_drawdown: float
    
    @validator('mode')
    def validate_mode(cls, v):
        if v not in ['paper', 'live']:
            raise ValueError('mode must be "paper" or "live"')
        return v
    
    @validator('initial_capital', 'max_position_size', 'max_daily_loss')
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError('must be positive')
        return v
    
    @validator('max_portfolio_risk', 'max_drawdown')
    def validate_ratio(cls, v):
        if not 0 < v <= 1:
            raise ValueError('must be between 0 and 1')
        return v


class BrokerConfig(BaseModel):
    """Broker configuration validation"""
    name: str
    paper_url: str
    live_url: str
    timeout: int
    
    @validator('timeout')
    def validate_timeout(cls, v):
        if v <= 0:
            raise ValueError('timeout must be positive')
        return v


class SystemConfig(BaseModel):
    """System configuration validation"""
    name: str
    version: str
    environment: str
    log_level: str
    timezone: str
    
    @validator('environment')
    def validate_environment(cls, v):
        if v not in ['development', 'staging', 'production']:
            raise ValueError('environment must be development, staging, or production')
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        if v not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            raise ValueError('invalid log level')
        return v


class ConfigValidator:
    """Main configuration validator"""
    
    def __init__(self, config_path: str = "config/config.yaml", env_file: str = ".env"):
        self.config_path = Path(config_path)
        self.env_file = Path(env_file)
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def validate_all(self) -> bool:
        """Run all validation checks"""
        print("🔍 Validating QuantumSentiment configuration...")
        print("=" * 50)
        
        success = True
        
        # Check file existence
        success &= self._check_files_exist()
        
        # Load environment variables
        if self.env_file.exists():
            load_dotenv(self.env_file)
        
        # Load config
        config = self._load_config()
        if not config:
            success = False
        else:
            # Validate configuration sections
            success &= self._validate_config_structure(config)
            success &= self._validate_environment_variables()
            success &= self._validate_api_credentials()
            success &= self._validate_safety_settings()
            success &= self._validate_directories()
            
        self._print_results()
        return success
    
    def _check_files_exist(self) -> bool:
        """Check if required files exist"""
        print("📁 Checking required files...")
        
        success = True
        required_files = [
            "requirements.txt",
            "requirements-ml.txt", 
            "pyproject.toml",
            ".env.example"
        ]
        
        for file in required_files:
            if not Path(file).exists():
                self.errors.append(f"Missing required file: {file}")
                success = False
            else:
                print(f"  ✅ {file}")
        
        # Check config file
        if not self.config_path.exists():
            if Path("config/config.example.yaml").exists():
                self.warnings.append(f"Config file {self.config_path} not found. Copy from config.example.yaml")
            else:
                self.errors.append(f"Config file {self.config_path} not found")
                success = False
        else:
            print(f"  ✅ {self.config_path}")
            
        # Check env file
        if not self.env_file.exists():
            self.warnings.append(f"Environment file {self.env_file} not found. Copy from .env.example")
        else:
            print(f"  ✅ {self.env_file}")
            
        return success
    
    def _load_config(self) -> Optional[Dict[str, Any]]:
        """Load and parse YAML configuration"""
        if not self.config_path.exists():
            return None
            
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"  ✅ Loaded config from {self.config_path}")
            return config
        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML in {self.config_path}: {e}")
            return None
        except Exception as e:
            self.errors.append(f"Error loading {self.config_path}: {e}")
            return None
    
    def _validate_config_structure(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure using Pydantic models"""
        print("⚙️  Validating configuration structure...")
        
        success = True
        
        # Validate system config
        try:
            SystemConfig(**config.get('system', {}))
            print("  ✅ System configuration")
        except ValidationError as e:
            self.errors.append(f"System config validation failed: {e}")
            success = False
        
        # Validate trading config
        try:
            TradingConfig(**config.get('trading', {}))
            print("  ✅ Trading configuration")
        except ValidationError as e:
            self.errors.append(f"Trading config validation failed: {e}")
            success = False
        
        # Validate broker config
        try:
            BrokerConfig(**config.get('broker', {}))
            print("  ✅ Broker configuration")
        except ValidationError as e:
            self.errors.append(f"Broker config validation failed: {e}")
            success = False
        
        return success
    
    def _validate_environment_variables(self) -> bool:
        """Validate required environment variables"""
        print("🔐 Validating environment variables...")
        
        success = True
        required_vars = [
            'ALPACA_API_KEY',
            'ALPACA_API_SECRET',
            'ALPACA_BASE_URL',
            'REDDIT_CLIENT_ID',
            'REDDIT_CLIENT_SECRET',
            'REDDIT_USER_AGENT',
            'ALPHA_VANTAGE_API_KEY',
        ]
        
        for var in required_vars:
            value = os.getenv(var)
            if not value or value.startswith('your_'):
                self.errors.append(f"Missing or placeholder value for {var}")
                success = False
            else:
                print(f"  ✅ {var}")
        
        # Check safety variables
        trading_mode = os.getenv('TRADING_MODE', 'paper')
        if trading_mode == 'live':
            confirm = os.getenv('CONFIRM_REAL_MONEY_TRADING', 'false').lower()
            if confirm != 'true':
                self.errors.append("Live trading requires CONFIRM_REAL_MONEY_TRADING=true")
                success = False
        
        return success
    
    def _validate_api_credentials(self) -> bool:
        """Test API credentials if possible"""
        print("🌐 Testing API connections...")
        
        success = True
        
        # Test Alpaca connection
        try:
            import alpaca_trade_api as tradeapi
            api = tradeapi.REST(
                os.getenv('ALPACA_API_KEY'),
                os.getenv('ALPACA_SECRET_KEY'),
                os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
                api_version='v2'
            )
            account = api.get_account()
            print(f"  ✅ Alpaca API ({account.status})")
        except Exception as e:
            self.warnings.append(f"Could not test Alpaca API: {e}")
        
        # Test Reddit API
        try:
            import praw
            reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent=os.getenv('REDDIT_USER_AGENT')
            )
            # Test read-only access
            reddit.subreddit('wallstreetbets').hot(limit=1)
            print("  ✅ Reddit API")
        except Exception as e:
            self.warnings.append(f"Could not test Reddit API: {e}")
        
        # Test Alpha Vantage
        try:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={os.getenv('ALPHA_VANTAGE_API_KEY')}"
            response = requests.get(url, timeout=10)
            data = response.json()
            if 'Error Message' in data:
                self.warnings.append(f"Alpha Vantage API error: {data['Error Message']}")
            elif 'Note' in data:
                self.warnings.append(f"Alpha Vantage rate limit: {data['Note']}")
            else:
                print("  ✅ Alpha Vantage API")
        except Exception as e:
            self.warnings.append(f"Could not test Alpha Vantage API: {e}")
        
        return success
    
    def _validate_safety_settings(self) -> bool:
        """Validate safety and risk settings"""
        print("🛡️  Validating safety settings...")
        
        success = True
        
        # Check position limits
        max_position = float(os.getenv('MAX_POSITION_SIZE_EUR', 50))
        max_daily_loss = float(os.getenv('MAX_DAILY_LOSS_EUR', 100))
        
        if max_position > 100:
            self.warnings.append(f"Max position size {max_position}€ is quite high for €1000 capital")
        
        if max_daily_loss > 200:
            self.warnings.append(f"Max daily loss {max_daily_loss}€ is high for €1000 capital")
        
        # Check trading mode
        trading_mode = os.getenv('TRADING_MODE', 'paper')
        if trading_mode == 'paper':
            print("  ✅ Paper trading mode (safe)")
        else:
            print("  ⚠️  Live trading mode - ensure this is intentional!")
        
        return success
    
    def _validate_directories(self) -> bool:
        """Validate required directories exist"""
        print("📂 Checking directory structure...")
        
        success = True
        required_dirs = [
            "src",
            "config", 
            "scripts",
            "tests",
            "data",
            "models",
            "logs"
        ]
        
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    print(f"  ✅ Created {dir_name}/")
                except Exception as e:
                    self.errors.append(f"Could not create directory {dir_name}: {e}")
                    success = False
            else:
                print(f"  ✅ {dir_name}/")
        
        return success
    
    def _print_results(self):
        """Print validation results"""
        print("\n" + "=" * 50)
        print("📋 VALIDATION RESULTS")
        print("=" * 50)
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
            print("\n🚨 Configuration validation FAILED!")
            print("   Please fix the errors above before running the trading bot.")
        else:
            print("\n✅ Configuration validation PASSED!")
            print("   Your QuantumSentiment bot is ready to run.")
            
        print("\n💡 Next steps:")
        if self.errors:
            print("   1. Fix the configuration errors listed above")
            print("   2. Re-run this validation script")
        else:
            print("   1. Install dependencies: uv pip install -r requirements.txt")
            print("   2. Install ML dependencies: uv pip install -r requirements-ml.txt")  
            print("   3. Run the bot: python src/main.py --mode paper")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Validate QuantumSentiment configuration")
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    parser.add_argument('--env-file', default='.env', help='Path to environment file')
    parser.add_argument('--quiet', action='store_true', help='Only show errors')
    
    args = parser.parse_args()
    
    if args.quiet:
        # Redirect stdout for quiet mode
        import io
        sys.stdout = io.StringIO()
    
    validator = ConfigValidator(args.config, args.env_file)
    success = validator.validate_all()
    
    if args.quiet:
        sys.stdout = sys.__stdout__
        if not success:
            print("Configuration validation failed. Run without --quiet for details.")
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()