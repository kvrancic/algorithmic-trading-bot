#!/usr/bin/env python3
"""
Setup Verification Script

Checks that all components are properly configured and working.
Run this before starting the trading bot for the first time.
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import importlib.util

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


class SetupVerifier:
    """Verifies trading bot setup and configuration"""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []
        
    def print_header(self, text: str) -> None:
        """Print section header"""
        print(f"\n{BLUE}{'='*60}")
        print(f"  {text}")
        print(f"{'='*60}{RESET}")
    
    def print_check(self, name: str, passed: bool, message: str = "") -> None:
        """Print check result"""
        if passed:
            status = f"{GREEN}✓ PASS{RESET}"
            self.checks_passed += 1
        else:
            status = f"{RED}✗ FAIL{RESET}"
            self.checks_failed += 1
        
        print(f"  {status} {name}")
        if message:
            print(f"       {message}")
    
    def print_warning(self, message: str) -> None:
        """Print warning message"""
        print(f"  {YELLOW}⚠ WARNING{RESET} {message}")
        self.warnings.append(message)
    
    async def check_python_version(self) -> bool:
        """Check Python version"""
        self.print_header("Python Version Check")
        
        version = sys.version_info
        required = (3, 9)
        
        passed = version >= required
        self.print_check(
            "Python Version",
            passed,
            f"Current: {version.major}.{version.minor}.{version.micro}, Required: {required[0]}.{required[1]}+"
        )
        
        return passed
    
    async def check_dependencies(self) -> bool:
        """Check required dependencies"""
        self.print_header("Dependencies Check")
        
        required_packages = [
            'pandas',
            'numpy',
            'structlog',
            'alpaca_trade_api',
            'asyncio',
            'aiohttp',
            'sqlalchemy',
            ('sklearn', 'scikit-learn'),  # Handle sklearn/scikit-learn naming
            'xgboost',
            'tensorflow',
            'torch',
            'transformers',
            'praw',  # Reddit API
            'vaderSentiment',
            'textblob',
            'yfinance',
            'ta',  # Technical indicators
            'scipy',
            'matplotlib',
            'plotly'
        ]
        
        all_passed = True
        
        for package in required_packages:
            try:
                if isinstance(package, tuple):
                    # Try alternative names
                    package_name, display_name = package
                    try:
                        importlib.import_module(package_name)
                    except ImportError:
                        importlib.import_module('sklearn')
                    self.print_check(display_name, True)
                elif package == 'alpaca_trade_api':
                    importlib.import_module('alpaca_trade_api')
                    self.print_check(package, True)
                else:
                    importlib.import_module(package)
                    self.print_check(package, True)
            except ImportError:
                display_name = package[1] if isinstance(package, tuple) else package
                self.print_check(display_name, False, "Not installed")
                all_passed = False
        
        return all_passed
    
    async def check_environment_variables(self) -> bool:
        """Check environment variables"""
        self.print_header("Environment Variables Check")
        
        # Check for .env file
        env_file = Path('.env')
        if not env_file.exists():
            self.print_check(".env file", False, "File not found")
            return False
        else:
            self.print_check(".env file", True)
        
        # Load .env file
        from dotenv import load_dotenv
        load_dotenv()
        
        # Required variables for paper trading
        required_vars = [
            'ALPACA_API_KEY',
            'ALPACA_API_SECRET'
        ]
        
        # Optional variables
        optional_vars = [
            'DATABASE_URL',
            'REDDIT_CLIENT_ID',
            'REDDIT_CLIENT_SECRET',
            'NEWSAPI_KEY',
            'ALPHA_VANTAGE_API_KEY'
        ]
        
        all_required = True
        
        # Check required variables
        for var in required_vars:
            value = os.getenv(var)
            if value:
                self.print_check(f"{var}", True, "Set")
            else:
                self.print_check(f"{var}", False, "Not set")
                all_required = False
        
        # Check optional variables
        for var in optional_vars:
            value = os.getenv(var)
            if not value:
                self.print_warning(f"{var} not set (optional)")
        
        return all_required
    
    async def check_configuration(self) -> bool:
        """Check configuration files"""
        self.print_header("Configuration Files Check")
        
        config_file = Path('config/config.yaml')
        
        if not config_file.exists():
            self.print_check("config.yaml", False, "File not found")
            
            # Check for example file
            example_file = Path('config/config.example.yaml')
            if example_file.exists():
                self.print_warning("Found config.example.yaml - copy to config.yaml")
            
            return False
        
        self.print_check("config.yaml", True)
        
        # Try to load and validate config
        try:
            import yaml
            with open(config_file) as f:
                config = yaml.safe_load(f)
            
            # Check essential config sections
            sections = ['trading', 'risk', 'database']
            for section in sections:
                if section in config:
                    self.print_check(f"Config section: {section}", True)
                else:
                    self.print_check(f"Config section: {section}", False, "Missing")
            
            return all(section in config for section in sections)
            
        except Exception as e:
            self.print_check("Config validation", False, str(e))
            return False
    
    async def check_database(self) -> bool:
        """Check database connectivity"""
        self.print_header("Database Check")
        
        try:
            from src.database import DatabaseManager
            
            db_url = os.getenv('DATABASE_URL', 'sqlite:///trading_bot.db')
            
            # Test connection
            db = DatabaseManager(db_url)
            
            # Check if database is accessible
            self.print_check("Database connection", True, f"Connected to {db_url.split(':')[0]} database")
            
            self.print_warning("Database tables will be created on first run")
            
            return True
            
        except Exception as e:
            self.print_check("Database connection", False, str(e))
            return False
    
    async def check_alpaca_connection(self) -> bool:
        """Check Alpaca API connection"""
        self.print_header("Alpaca API Check")
        
        try:
            from src.data.alpaca_client import AlpacaClient
            
            # Check paper trading connection
            paper_key = os.getenv('ALPACA_API_KEY')
            paper_secret = os.getenv('ALPACA_API_SECRET')
            
            if not paper_key or not paper_secret:
                self.print_check("Alpaca credentials", False, "API keys not found")
                return False
            
            client = AlpacaClient(
                api_key=paper_key,
                api_secret=paper_secret,
                paper=True
            )
            
            # Test account access
            account = client.get_account()
            if account:
                self.print_check("Alpaca connection", True, "Paper trading account active")
                self.print_check("Account status", True, f"Equity: ${float(account.equity):,.2f}")
                
                # Check market data access
                try:
                    quote = client.get_latest_quote("SPY")
                    if quote:
                        self.print_check("Market data access", True)
                    else:
                        self.print_check("Market data access", False, "No data returned")
                except:
                    self.print_warning("Market data access failed - check subscription")
                
                return True
            else:
                self.print_check("Alpaca connection", False, "Could not access account")
                return False
                
        except Exception as e:
            self.print_check("Alpaca connection", False, str(e))
            return False
    
    async def check_directory_structure(self) -> bool:
        """Check required directories exist"""
        self.print_header("Directory Structure Check")
        
        required_dirs = [
            'src',
            'config',
            'logs',
            'data',
            'models',
            'scripts',
            'tests'
        ]
        
        all_exist = True
        
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists():
                self.print_check(f"Directory: {dir_name}", True)
            else:
                self.print_check(f"Directory: {dir_name}", False, "Creating...")
                dir_path.mkdir(exist_ok=True)
                all_exist = False
        
        return all_exist
    
    async def check_optional_services(self) -> None:
        """Check optional services"""
        self.print_header("Optional Services Check")
        
        # Reddit API
        reddit_id = os.getenv('REDDIT_CLIENT_ID')
        if reddit_id:
            try:
                import praw
                self.print_check("Reddit API", True, "Credentials found")
            except:
                self.print_warning("Reddit API credentials found but praw not installed")
        else:
            self.print_warning("Reddit API not configured (sentiment analysis limited)")
        
        # News APIs
        news_api = os.getenv('NEWSAPI_KEY')
        if news_api:
            self.print_check("News API", True, "Key found")
        else:
            self.print_warning("News API not configured (news sentiment unavailable)")
    
    def print_summary(self) -> None:
        """Print verification summary"""
        self.print_header("Verification Summary")
        
        total_checks = self.checks_passed + self.checks_failed
        
        print(f"\n  Total checks: {total_checks}")
        print(f"  {GREEN}Passed: {self.checks_passed}{RESET}")
        print(f"  {RED}Failed: {self.checks_failed}{RESET}")
        
        if self.warnings:
            print(f"\n  {YELLOW}Warnings: {len(self.warnings)}{RESET}")
            for warning in self.warnings[:5]:  # Show first 5 warnings
                print(f"  - {warning}")
        
        if self.checks_failed == 0:
            print(f"\n{GREEN}✓ All checks passed! System is ready for trading.{RESET}")
            print(f"\nNext step: Run {BLUE}python -m src.main --mode paper{RESET}")
        else:
            print(f"\n{RED}✗ Some checks failed. Please fix the issues above.{RESET}")
            print(f"\nFor help, see: {BLUE}docs/guides/getting_started.md{RESET}")
    
    async def run_all_checks(self) -> None:
        """Run all verification checks"""
        print(f"{BLUE}QuantumSentiment Trading Bot - Setup Verification{RESET}")
        print(f"{BLUE}================================================{RESET}")
        
        # Essential checks
        await self.check_python_version()
        await self.check_dependencies()
        await self.check_environment_variables()
        await self.check_configuration()
        await self.check_directory_structure()
        await self.check_database()
        await self.check_alpaca_connection()
        
        # Optional checks
        await self.check_optional_services()
        
        # Summary
        self.print_summary()


async def main():
    """Main entry point"""
    verifier = SetupVerifier()
    await verifier.run_all_checks()


if __name__ == "__main__":
    asyncio.run(main())