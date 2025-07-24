#!/usr/bin/env python3
"""
Historical Data Download Script for QuantumSentiment Trading Bot

Downloads and processes historical data for backtesting and model training:
- Market data (OHLCV) from multiple timeframes
- Sentiment data from Reddit, Twitter, News
- Fundamental data and economic indicators
- Validates, cleans, and stores data in database
- Generates comprehensive feature sets

Usage:
    python scripts/download_historical_data.py --symbols AAPL,TSLA,BTC-USD --days 365
    python scripts/download_historical_data.py --config config/download_config.yaml
"""

import argparse
import asyncio
import yaml
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import structlog
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.data_fetcher import DataFetcher, FetcherConfig
from data.data_interface import DataInterface
from database.database import DatabaseManager
from features.feature_pipeline import FeaturePipeline, FeatureConfig
from validation.data_validator import DataValidator, ValidationConfig
from validation.data_cleaner import DataCleaner, CleaningConfig

logger = structlog.get_logger(__name__)


class HistoricalDataDownloader:
    """Comprehensive historical data downloader"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize historical data downloader
        
        Args:
            config_file: Path to configuration file
        """
        self.config = self._load_config(config_file)
        
        # Initialize components
        self.data_interface = DataInterface()
        fetcher_config = FetcherConfig(max_workers=self.config.get('concurrent_downloads', 4))
        self.data_fetcher = DataFetcher(fetcher_config, self.data_interface)
        # Initialize database manager
        db_config = self.config.get('database', {})
        if isinstance(db_config, dict):
            db_url = db_config.get('url', 'sqlite:///quantum_sentiment.db')
        else:
            db_url = str(db_config)
        self.db_manager = DatabaseManager(db_url)
        
        # Initialize validation and cleaning
        validation_config = ValidationConfig(**self.config.get('validation', {}))
        cleaning_config = CleaningConfig(**self.config.get('cleaning', {}))
        
        self.validator = DataValidator(validation_config)
        self.cleaner = DataCleaner(cleaning_config)
        
        # Initialize feature pipeline
        feature_config = FeatureConfig(**self.config.get('features', {}))
        self.feature_pipeline = FeaturePipeline(feature_config, self.db_manager)
        
        # Download statistics
        self.stats = {
            'total_symbols': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_records': 0,
            'validation_failures': 0,
            'features_generated': 0,
            'start_time': datetime.now(),
            'errors': []
        }
        
        logger.info("Historical data downloader initialized", 
                   symbols=len(self.config.get('symbols', [])),
                   days=self.config.get('days', 365))
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'symbols': ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'BTC-USD', 'ETH-USD'],
            'days': 365,
            'timeframes': ['1min', '5min', '15min', '1hour', '1day'],
            'download_market_data': True,
            'download_sentiment_data': True,
            'download_fundamental_data': True,
            'generate_features': True,
            'validate_data': True,
            'clean_data': True,
            'concurrent_downloads': 4,
            'batch_size': 1000,
            'save_raw_data': True,
            'database': {
                'url': 'sqlite:///quantum_sentiment.db',
                'echo': False
            },
            'validation': {
                'strict_mode': False,
                'allow_gaps': True,
                'max_gap_hours': 24.0
            },
            'cleaning': {
                'handle_missing_values': True,
                'handle_outliers': True,
                'fix_ohlc_relationships': True
            },
            'features': {
                'enable_technical': True,
                'enable_sentiment': True,
                'parallel_processing': True,
                'enable_caching': False  # Disable caching for bulk download
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    # Merge configurations
                    default_config.update(file_config)
                logger.info("Configuration loaded", config_file=config_file)
            except Exception as e:
                logger.warning("Failed to load config file, using defaults", 
                             config_file=config_file, error=str(e))
        
        return default_config
    
    async def download_all_data(self, symbols: Optional[List[str]] = None, days: Optional[int] = None):
        """
        Download all historical data for specified symbols
        
        Args:
            symbols: List of symbols to download (overrides config)
            days: Number of days to download (overrides config)
        """
        symbols = symbols or self.config['symbols']
        days = days or self.config['days']
        
        self.stats['total_symbols'] = len(symbols)
        
        logger.info("Starting historical data download", 
                   symbols=len(symbols), days=days)
        
        try:
            # Initialize database
            self.db_manager.create_tables()
            
            # Process symbols in batches
            batch_size = self.config.get('concurrent_downloads', 4)
            
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                
                # Process batch concurrently
                tasks = [
                    self._download_symbol_data(symbol, days)
                    for symbol in batch_symbols
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for symbol, result in zip(batch_symbols, results):
                    if isinstance(result, Exception):
                        self.stats['failed_downloads'] += 1
                        self.stats['errors'].append(f"{symbol}: {str(result)}")
                        logger.error("Symbol download failed", 
                                   symbol=symbol, error=str(result))
                    else:
                        self.stats['successful_downloads'] += 1
                        logger.info("Symbol download completed", 
                                  symbol=symbol, records=result.get('total_records', 0))
            
            # Generate final statistics
            await self._generate_final_report()
            
        except Exception as e:
            logger.error("Download process failed", error=str(e))
            raise
        finally:
            # DatabaseManager doesn't have async close method
            pass
    
    async def _download_symbol_data(self, symbol: str, days: int) -> Dict[str, Any]:
        """Download all data for a single symbol"""
        symbol_stats = {
            'symbol': symbol,
            'total_records': 0,
            'market_data_records': 0,
            'sentiment_data_records': 0,
            'features_generated': 0,
            'validation_passed': True,
            'processing_time': 0
        }
        
        start_time = datetime.now()
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            logger.info("Processing symbol", symbol=symbol, 
                       start_date=start_date.date(), end_date=end_date.date())
            
            # Download market data
            market_data = pd.DataFrame()
            if self.config['download_market_data']:
                market_data = await self._download_market_data(symbol, start_date, end_date, days)
                symbol_stats['market_data_records'] = len(market_data)
                symbol_stats['total_records'] += len(market_data)
            
            # Download sentiment data
            sentiment_data = pd.DataFrame()
            if self.config['download_sentiment_data']:
                sentiment_data = await self._download_sentiment_data(symbol, start_date, end_date, days)
                symbol_stats['sentiment_data_records'] = len(sentiment_data)
                symbol_stats['total_records'] += len(sentiment_data)
            
            # Download fundamental data
            fundamental_data = {}
            if self.config['download_fundamental_data']:
                fundamental_data = await self._download_fundamental_data(symbol)
            
            # Validate data
            if self.config['validate_data']:
                validation_passed = await self._validate_symbol_data(
                    symbol, market_data, sentiment_data
                )
                symbol_stats['validation_passed'] = validation_passed
                
                if not validation_passed:
                    self.stats['validation_failures'] += 1
            
            # Clean data
            if self.config['clean_data']:
                market_data, sentiment_data = await self._clean_symbol_data(
                    symbol, market_data, sentiment_data
                )
            
            # Generate features
            if self.config['generate_features'] and not market_data.empty:
                features_count = await self._generate_features(
                    symbol, market_data, sentiment_data, fundamental_data
                )
                symbol_stats['features_generated'] = features_count
                self.stats['features_generated'] += features_count
            
            # Store data in database
            await self._store_symbol_data(
                symbol, market_data, sentiment_data, fundamental_data
            )
            
            # Update global statistics
            self.stats['total_records'] += symbol_stats['total_records']
            
        except Exception as e:
            logger.error("Symbol processing failed", symbol=symbol, error=str(e))
            raise
        
        symbol_stats['processing_time'] = (datetime.now() - start_time).total_seconds()
        return symbol_stats
    
    async def _download_market_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        days: int
    ) -> pd.DataFrame:
        """Download market data for symbol"""
        try:
            all_market_data = []
            
            for timeframe in self.config['timeframes']:
                logger.debug("Downloading market data", 
                           symbol=symbol, timeframe=timeframe)
                
                # Get data from data fetcher
                market_results = await self.data_fetcher.fetch_market_data(
                    symbols=[symbol],
                    timeframe=timeframe,
                    days_back=days
                )
                data = market_results.get(symbol, pd.DataFrame())
                
                if not data.empty:
                    data['timeframe'] = timeframe
                    all_market_data.append(data)
            
            if all_market_data:
                combined_data = pd.concat(all_market_data, ignore_index=True)
                logger.debug("Market data downloaded", 
                           symbol=symbol, records=len(combined_data))
                return combined_data
            else:
                logger.warning("No market data found", symbol=symbol)
                return pd.DataFrame()
                
        except Exception as e:
            logger.error("Market data download failed", 
                        symbol=symbol, error=str(e))
            return pd.DataFrame()
    
    async def _download_sentiment_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        days: int
    ) -> pd.DataFrame:
        """Download sentiment data for symbol"""
        try:
            all_sentiment_data = []
            
            # Get sentiment data from all sources
            sources = ['reddit', 'twitter', 'news', 'unusual_whales']
            
            for source in sources:
                logger.debug("Downloading sentiment data", 
                           symbol=symbol, source=source)
                
                sentiment_result = await self.data_fetcher.fetch_sentiment_data(
                    symbols=[symbol],
                    sources=[source],
                    hours_back=days * 24  # Convert days to hours
                )
                
                # Convert dict result to DataFrame if we got data
                if sentiment_result and symbol in sentiment_result:
                    symbol_sentiment = sentiment_result[symbol]
                    if symbol_sentiment and 'data' in symbol_sentiment:
                        data = pd.DataFrame(symbol_sentiment['data'])
                        if not data.empty:
                            data['source'] = source
                            all_sentiment_data.append(data)
            
            if all_sentiment_data:
                combined_data = pd.concat(all_sentiment_data, ignore_index=True)
                logger.debug("Sentiment data downloaded", 
                           symbol=symbol, records=len(combined_data))
                return combined_data
            else:
                logger.debug("No sentiment data found", symbol=symbol)
                return pd.DataFrame()
                
        except Exception as e:
            logger.error("Sentiment data download failed", 
                        symbol=symbol, error=str(e))
            return pd.DataFrame()
    
    async def _download_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """Download fundamental data for symbol"""
        try:
            logger.debug("Downloading fundamental data", symbol=symbol)
            
            fundamental_data = await self.data_fetcher.fetch_fundamental_data([symbol])
            fundamental_data = fundamental_data.get(symbol, {})
            
            logger.debug("Fundamental data downloaded", 
                        symbol=symbol, fields=len(fundamental_data))
            return fundamental_data
            
        except Exception as e:
            logger.error("Fundamental data download failed", 
                        symbol=symbol, error=str(e))
            return {}
    
    async def _validate_symbol_data(
        self, 
        symbol: str, 
        market_data: pd.DataFrame, 
        sentiment_data: pd.DataFrame
    ) -> bool:
        """Validate downloaded data"""
        try:
            validation_passed = True
            
            # Validate market data
            if not market_data.empty:
                market_validation = self.validator.validate_market_data(
                    market_data, symbol
                )
                if not market_validation.is_valid:
                    logger.warning("Market data validation failed", 
                                 symbol=symbol, 
                                 issues=len(market_validation.issues))
                    validation_passed = False
            
            # Validate sentiment data
            if not sentiment_data.empty:
                sentiment_validation = self.validator.validate_sentiment_data(
                    sentiment_data, symbol
                )
                if not sentiment_validation.is_valid:
                    logger.warning("Sentiment data validation failed", 
                                 symbol=symbol, 
                                 issues=len(sentiment_validation.issues))
                    validation_passed = False
            
            return validation_passed
            
        except Exception as e:
            logger.error("Data validation failed", symbol=symbol, error=str(e))
            return False
    
    async def _clean_symbol_data(
        self, 
        symbol: str, 
        market_data: pd.DataFrame, 
        sentiment_data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Clean downloaded data"""
        try:
            cleaned_market_data = market_data
            cleaned_sentiment_data = sentiment_data
            
            # Clean market data
            if not market_data.empty:
                cleaned_market_data, market_report = self.cleaner.clean_market_data(
                    market_data, symbol=symbol
                )
                logger.debug("Market data cleaned", 
                           symbol=symbol, 
                           operations=len(market_report.get('operations_performed', [])))
            
            # Clean sentiment data
            if not sentiment_data.empty:
                cleaned_sentiment_data, sentiment_report = self.cleaner.clean_sentiment_data(
                    sentiment_data, symbol=symbol
                )
                logger.debug("Sentiment data cleaned", 
                           symbol=symbol, 
                           operations=len(sentiment_report.get('operations_performed', [])))
            
            return cleaned_market_data, cleaned_sentiment_data
            
        except Exception as e:
            logger.error("Data cleaning failed", symbol=symbol, error=str(e))
            return market_data, sentiment_data
    
    async def _generate_features(
        self, 
        symbol: str, 
        market_data: pd.DataFrame, 
        sentiment_data: pd.DataFrame,
        fundamental_data: Dict[str, Any]
    ) -> int:
        """Generate features for symbol"""
        try:
            if market_data.empty:
                return 0
            
            logger.debug("Generating features", symbol=symbol)
            
            # Generate features for each day
            features_generated = 0
            
            # Group market data by date
            if 'timestamp' in market_data.columns:
                market_data['date'] = pd.to_datetime(market_data['timestamp']).dt.date
                
                for date, day_data in market_data.groupby('date'):
                    # Get sentiment data for this date
                    date_sentiment = sentiment_data[
                        pd.to_datetime(sentiment_data['timestamp']).dt.date == date
                    ] if not sentiment_data.empty and 'timestamp' in sentiment_data.columns else pd.DataFrame()
                    
                    # Generate features
                    result = self.feature_pipeline.generate_features(
                        symbol=symbol,
                        market_data=day_data,
                        sentiment_data=date_sentiment,
                        fundamental_data=fundamental_data,
                        current_time=datetime.combine(date, datetime.min.time()),
                        use_cache=False
                    )
                    
                    if result['features']:
                        features_generated += len(result['features'])
                        
                        # Store features in database
                        await self._store_features(symbol, date, result)
            
            logger.debug("Features generated", 
                        symbol=symbol, count=features_generated)
            return features_generated
            
        except Exception as e:
            logger.error("Feature generation failed", symbol=symbol, error=str(e))
            return 0
    
    async def _store_symbol_data(
        self, 
        symbol: str, 
        market_data: pd.DataFrame, 
        sentiment_data: pd.DataFrame,
        fundamental_data: Dict[str, Any]
    ):
        """Store all data for symbol in database"""
        try:
            # Store market data
            if not market_data.empty:
                for _, row in market_data.iterrows():
                    ohlcv_data = {
                        'open': row.get('open', 0),
                        'high': row.get('high', 0),
                        'low': row.get('low', 0),
                        'close': row.get('close', 0),
                        'volume': row.get('volume', 0)
                    }
                    self.db_manager.save_market_data(
                        symbol=symbol,
                        timestamp=row.get('timestamp', datetime.now()),
                        timeframe=row.get('timeframe', '1day'),
                        ohlcv_data=ohlcv_data
                    )
                logger.debug("Market data stored", 
                           symbol=symbol, records=len(market_data))
            
            # Store sentiment data
            if not sentiment_data.empty:
                for _, row in sentiment_data.iterrows():
                    self.db_manager.save_sentiment_data(
                        symbol=symbol,
                        timestamp=row.get('timestamp', datetime.now()),
                        source=row.get('source', 'unknown'),
                        sentiment_score=row.get('sentiment_score', 0),
                        confidence=row.get('confidence', 0),
                        mention_count=row.get('mention_count', 0),
                        raw_data=row.get('raw_data', {})
                    )
                logger.debug("Sentiment data stored", 
                           symbol=symbol, records=len(sentiment_data))
            
            # Store fundamental data (simplified)
            if fundamental_data:
                logger.debug("Fundamental data available", 
                           symbol=symbol, fields=len(fundamental_data))
            
        except Exception as e:
            logger.error("Data storage failed", symbol=symbol, error=str(e))
            raise
    
    async def _store_features(self, symbol: str, date, features_result: Dict[str, Any]):
        """Store generated features in database"""
        try:
            if features_result['features']:
                self.db_manager.save_feature_data(
                    symbol=symbol,
                    timestamp=datetime.combine(date, datetime.min.time()),
                    features=features_result['features']
                )
                
        except Exception as e:
            logger.error("Feature storage failed", 
                        symbol=symbol, date=date, error=str(e))
    
    async def _generate_final_report(self):
        """Generate and display final download report"""
        end_time = datetime.now()
        total_time = (end_time - self.stats['start_time']).total_seconds()
        
        report = {
            'summary': {
                'total_symbols': self.stats['total_symbols'],
                'successful_downloads': self.stats['successful_downloads'],
                'failed_downloads': self.stats['failed_downloads'],
                'success_rate': self.stats['successful_downloads'] / max(self.stats['total_symbols'], 1),
                'total_records': self.stats['total_records'],
                'features_generated': self.stats['features_generated'],
                'validation_failures': self.stats['validation_failures'],
                'total_processing_time': f"{total_time:.2f} seconds",
                'records_per_second': self.stats['total_records'] / max(total_time, 1)
            },
            'errors': self.stats['errors'][:10],  # Show first 10 errors
            'validation_stats': self.validator.get_validation_stats(),
            'cleaning_stats': self.cleaner.get_cleaning_stats(),
            'pipeline_stats': self.feature_pipeline.get_pipeline_stats()
        }
        
        logger.info("Historical data download completed", **report['summary'])
        
        # Save report to file
        report_file = f"download_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        with open(report_file, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        logger.info("Download report saved", report_file=report_file)
        
        # Print summary to console
        print("\n" + "="*80)
        print("HISTORICAL DATA DOWNLOAD REPORT")
        print("="*80)
        print(f"Total Symbols: {report['summary']['total_symbols']}")
        print(f"Successful Downloads: {report['summary']['successful_downloads']}")
        print(f"Failed Downloads: {report['summary']['failed_downloads']}")
        print(f"Success Rate: {report['summary']['success_rate']:.1%}")
        print(f"Total Records: {report['summary']['total_records']:,}")
        print(f"Features Generated: {report['summary']['features_generated']:,}")
        print(f"Processing Time: {report['summary']['total_processing_time']}")
        print(f"Records/Second: {report['summary']['records_per_second']:.1f}")
        
        if self.stats['errors']:
            print(f"\nFirst {len(report['errors'])} Errors:")
            for error in report['errors']:
                print(f"  - {error}")
        
        print("="*80)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Download historical trading data")
    
    parser.add_argument(
        '--symbols', '-s',
        type=str,
        help='Comma-separated list of symbols (e.g., AAPL,TSLA,BTC-USD)'
    )
    
    parser.add_argument(
        '--days', '-d',
        type=int,
        help='Number of days of historical data to download'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='data',
        help='Output directory for data files'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform dry run without actually downloading data'
    )
    
    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_args()
    
    # Configure logging
    import logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    try:
        # Initialize downloader
        downloader = HistoricalDataDownloader(args.config)
        
        # Parse symbols
        symbols = None
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(',')]
        
        if args.dry_run:
            logger.info("Dry run mode - no data will be downloaded")
            logger.info("Configuration loaded", 
                       symbols=symbols or downloader.config['symbols'],
                       days=args.days or downloader.config['days'])
            return
        
        # Start download
        await downloader.download_all_data(symbols=symbols, days=args.days)
        
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
    except Exception as e:
        logger.error("Download failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())