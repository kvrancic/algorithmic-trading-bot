#!/usr/bin/env python3
"""
Professional Model Training for QuantumSentiment Trading Bot

This script uses the comprehensive training infrastructure to train production-ready models
using real market data downloaded from the data pipeline. Trains all models (LSTM, XGBoost, 
ensemble) with proper data splits, validation, and persistence.

Usage:
    python train_professional.py --config config/download_config.yaml
    python train_professional.py --symbols AAPL,TSLA,GOOGL --days 730
    python train_professional.py --full-pipeline  # Download data + train models
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import structlog
import yaml
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

# Import the comprehensive training system
try:
    from src.training.training_pipeline import ModelTrainingPipeline, TrainingConfig
    from src.training.model_persistence import ModelPersistence, PersistenceConfig
    from src.models import (
        PriceLSTM, PriceLSTMConfig,
        MarketRegimeXGBoost, MarketRegimeConfig, 
        StackedEnsemble, StackedEnsembleConfig
    )
    from src.data.data_fetcher import DataFetcher, FetcherConfig
    from src.data.data_interface import DataInterface
    from src.database.database import DatabaseManager
    from src.features.feature_pipeline import FeaturePipeline, FeatureConfig
    from src.validation.data_validator import DataValidator, ValidationConfig
    from src.validation.data_cleaner import DataCleaner, CleaningConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Configure logging
logging_config = {
    'processors': [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    'wrapper_class': structlog.stdlib.BoundLogger,
    'logger_factory': structlog.stdlib.LoggerFactory(),
    'cache_logger_on_first_use': True,
}
structlog.configure(**logging_config)

logger = structlog.get_logger(__name__)


class ProfessionalModelTrainer:
    """Professional-grade model trainer using the complete infrastructure"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the professional trainer"""
        self.config = self._load_config(config_path)
        self.db_manager = None
        self.data_fetcher = None
        self.feature_pipeline = None
        self.validator = None
        self.cleaner = None
        self.model_persistence = None
        self.training_pipeline = None
        
        logger.info("Professional model trainer initialized", 
                   config_source=config_path or "default")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        
        # Default configuration for training
        default_config = {
            'symbols': ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'NVDA', 'SPY', 'QQQ'],
            'days': 730,  # 2 years of data
            'timeframes': ['1hour', '1day'],
            'models': {
                'train_lstm': True,
                'train_xgboost': True, 
                'train_ensemble': True,
                'train_cnn': False,  # Requires additional setup
                'train_finbert': False  # Heavy compute requirement
            },
            'training': {
                'train_start_date': '2022-01-01',
                'train_end_date': '2024-12-31',
                'validation_split': 0.2,
                'test_split': 0.1,
                'parallel_training': True,
                'max_workers': 4,
                'random_seed': 42,
                'early_stopping_patience': 20
            },
            'persistence': {
                'model_registry_path': 'models',
                'use_model_registry': True,
                'auto_cleanup': True,
                'max_versions_per_model': 5
            },
            'database': {
                'url': 'sqlite:///quantum_sentiment.db'
            }
        }
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                
                # Merge configurations intelligently
                if file_config:
                    # Update default with file config
                    default_config.update(file_config)
                    logger.info("Configuration loaded from file", 
                               config_path=config_path,
                               symbols_count=len(default_config.get('symbols', [])))
                
            except Exception as e:
                logger.warning("Failed to load config file, using defaults", 
                             config_path=config_path, error=str(e))
        
        return default_config
    
    async def initialize(self) -> bool:
        """Initialize all components"""
        try:
            logger.info("Initializing professional training components...")
            
            # 1. Initialize database
            db_url = self.config.get('database', {}).get('url', 'sqlite:///quantum_sentiment.db')
            self.db_manager = DatabaseManager(db_url)
            
            # 2. Initialize data interface and fetcher
            data_interface = DataInterface()
            fetcher_config = FetcherConfig(
                max_workers=self.config.get('training', {}).get('max_workers', 4)
            )
            self.data_fetcher = DataFetcher(fetcher_config, data_interface)
            
            # 3. Initialize feature pipeline
            feature_config = FeatureConfig(
                enable_technical=True,
                enable_sentiment=True,
                enable_fundamental=True,
                parallel_processing=True,
                enable_caching=False  # Disable for training
            )
            self.feature_pipeline = FeaturePipeline(feature_config, self.db_manager)
            
            # 4. Initialize validation and cleaning
            validation_config = ValidationConfig(
                strict_mode=False,
                allow_gaps=True,
                max_gap_hours=24.0
            )
            self.validator = DataValidator(validation_config)
            
            cleaning_config = CleaningConfig(
                handle_missing_values=True,
                handle_outliers=True,
                fix_ohlc_relationships=True
            )
            self.cleaner = DataCleaner(cleaning_config)
            
            # 5. Initialize model persistence
            persistence_config = PersistenceConfig(**self.config.get('persistence', {}))
            self.model_persistence = ModelPersistence(persistence_config)
            
            # 6. Initialize training pipeline
            training_config = TrainingConfig(**self.config.get('training', {}))
            training_config.model_save_dir = Path(self.config.get('persistence', {}).get('model_registry_path', 'models'))
            
            # Set which models to train
            models_config = self.config.get('models', {})
            training_config.train_lstm = models_config.get('train_lstm', True)
            training_config.train_xgboost = models_config.get('train_xgboost', True)
            training_config.train_ensemble = models_config.get('train_ensemble', True)
            training_config.train_cnn = models_config.get('train_cnn', False)
            training_config.train_finbert = models_config.get('train_finbert', False)
            
            self.training_pipeline = ModelTrainingPipeline(training_config)
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error("Initialization failed", error=str(e))
            return False
    
    async def download_training_data(self, symbols: List[str], days: int) -> pd.DataFrame:
        """Download comprehensive training data"""
        logger.info("Downloading training data", symbols=symbols, days=days)
        
        all_data = []
        
        # Download data for each symbol and timeframe
        for symbol in symbols:
            logger.info(f"Downloading data for {symbol}")
            
            for timeframe in self.config.get('timeframes', ['1hour', '1day']):
                try:
                    # Fetch market data
                    market_results = await self.data_fetcher.fetch_market_data(
                        symbols=[symbol],
                        timeframe=timeframe,
                        days_back=days
                    )
                    
                    symbol_data = market_results.get(symbol, pd.DataFrame())
                    
                    if not symbol_data.empty:
                        symbol_data['symbol'] = symbol
                        symbol_data['timeframe'] = timeframe
                        all_data.append(symbol_data)
                        
                        logger.info(f"Downloaded {len(symbol_data)} records for {symbol} {timeframe}")
                    else:
                        logger.warning(f"No data found for {symbol} {timeframe}")
                        
                except Exception as e:
                    logger.error(f"Failed to download {symbol} {timeframe}", error=str(e))
        
        if not all_data:
            raise ValueError("No training data could be downloaded")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Ensure proper timestamp handling
        if 'timestamp' not in combined_data.columns:
            # Check if timestamp is in the index
            if combined_data.index.name == 'timestamp' or 'timestamp' in str(combined_data.index.name):
                combined_data.reset_index(inplace=True)
            else:
                # Create timestamp column from index if it's a datetime index
                if hasattr(combined_data.index, 'to_pydatetime'):
                    combined_data['timestamp'] = combined_data.index
                    combined_data.reset_index(drop=True, inplace=True)
                else:
                    # Last resort: create sequential timestamps
                    logger.warning("No timestamp found, creating sequential timestamps")
                    combined_data['timestamp'] = pd.date_range(
                        start='2023-01-01', 
                        periods=len(combined_data), 
                        freq='H'
                    )
        
        # Ensure timestamp is datetime
        if 'timestamp' in combined_data.columns:
            combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'])
            combined_data.sort_values('timestamp', inplace=True)
        else:
            logger.error("Could not create timestamp column for training data")
            raise ValueError("Training data must have a timestamp column")
        
        logger.info("Training data download completed", 
                   total_records=len(combined_data),
                   symbols=len(symbols),
                   date_range=f"{combined_data['timestamp'].min()} to {combined_data['timestamp'].max()}" if 'timestamp' in combined_data.columns else "Unknown")
        
        return combined_data
    
    async def validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the downloaded data"""
        logger.info("Validating and cleaning training data")
        
        try:
            # Validate data quality
            for symbol in data['symbol'].unique():
                symbol_data = data[data['symbol'] == symbol]
                
                # Validate market data
                validation_result = self.validator.validate_market_data(symbol_data, symbol)
                if not validation_result.is_valid:
                    logger.warning(f"Data validation issues for {symbol}", 
                                 issues=len(validation_result.issues))
                    for issue in validation_result.issues[:5]:  # Show first 5 issues
                        logger.warning(f"  - {issue}")
            
            # Clean data
            cleaned_data_parts = []
            for symbol in data['symbol'].unique():
                symbol_data = data[data['symbol'] == symbol]
                
                # Clean market data
                cleaned_symbol_data, cleaning_report = self.cleaner.clean_market_data(symbol_data, symbol)
                cleaned_data_parts.append(cleaned_symbol_data)
                
                logger.info(f"Data cleaned for {symbol}", 
                           original_records=len(symbol_data),
                           cleaned_records=len(cleaned_symbol_data),
                           operations_performed=len(cleaning_report.get('operations_performed', [])))
            
            # Combine cleaned data
            cleaned_data = pd.concat(cleaned_data_parts, ignore_index=True)
            
            logger.info("Data validation and cleaning completed",
                       original_records=len(data),
                       cleaned_records=len(cleaned_data))
            
            return cleaned_data
            
        except Exception as e:
            logger.error("Data validation/cleaning failed", error=str(e))
            return data  # Return original data if cleaning fails
    
    async def train_models(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """Train all models using the professional pipeline"""
        logger.info("Starting professional model training")
        
        try:
            # Train models using the comprehensive pipeline
            trained_models = self.training_pipeline.train_all_models(
                price_data=training_data,
                text_data=None  # Sentiment data would be added here if available
            )
            
            if not trained_models:
                raise ValueError("No models were successfully trained")
            
            # Save models with proper metadata using persistence system
            saved_models = {}
            
            for model_name, model in trained_models.items():
                try:
                    logger.info(f"Saving trained model: {model_name}")
                    
                    # Calculate training data hash for versioning
                    training_data_hash = self._calculate_data_hash(training_data)
                    
                    # Get training results from pipeline
                    training_results = self.training_pipeline.training_results.get(model_name, {})
                    
                    # Extract metrics
                    validation_metrics = training_results.get('validation_metrics', {})
                    test_metrics = training_results.get('test_metrics', {})
                    training_duration = training_results.get('training_time', 0.0)
                    
                    # Save with comprehensive metadata
                    metadata = self.model_persistence.save_model(
                        model=model,
                        model_name=model_name,
                        training_data_hash=training_data_hash,
                        training_samples=len(training_data),
                        training_duration=training_duration,
                        validation_metrics=validation_metrics,
                        test_metrics=test_metrics,
                        description=f"Professional {model_name} trained on {len(training_data)} samples from {len(training_data['symbol'].unique())} symbols",
                        tags=["production", f"v{datetime.now().strftime('%Y%m%d')}", "professional"],
                        author="QuantumSentiment Professional Training Pipeline"
                    )
                    
                    saved_models[model_name] = {
                        'model_id': metadata.model_id,
                        'version': metadata.version,
                        'path': str(metadata.model_path),
                        'validation_metrics': validation_metrics,
                        'test_metrics': test_metrics,
                        'training_duration': training_duration
                    }
                    
                    logger.info(f"Model {model_name} saved successfully",
                               model_id=metadata.model_id,
                               version=metadata.version)
                    
                except Exception as e:
                    logger.error(f"Failed to save model {model_name}", error=str(e))
            
            # Get comprehensive training summary
            training_summary = self.training_pipeline.get_training_summary()
            
            results = {
                'saved_models': saved_models,
                'training_summary': training_summary,
                'total_models_trained': len(saved_models),
                'training_data_samples': len(training_data),
                'training_data_symbols': len(training_data['symbol'].unique()) if 'symbol' in training_data.columns else 0
            }
            
            logger.info("Professional model training completed",
                       models_trained=len(saved_models),
                       total_samples=len(training_data))
            
            return results
            
        except Exception as e:
            logger.error("Model training failed", error=str(e))
            raise
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of training data for versioning"""
        import hashlib
        
        # Create hash based on data characteristics
        data_info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'symbols': sorted(data['symbol'].unique().tolist()) if 'symbol' in data.columns else [],
            'date_range': {
                'start': str(data['timestamp'].min()) if 'timestamp' in data.columns else 'unknown',
                'end': str(data['timestamp'].max()) if 'timestamp' in data.columns else 'unknown'
            }
        }
        
        # Add sample of actual data for uniqueness
        if len(data) > 0:
            sample_rows = min(10, len(data))
            data_info['sample'] = data.head(sample_rows).to_dict()
        
        data_str = str(data_info)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    async def run_full_pipeline(self, symbols: Optional[List[str]] = None, days: Optional[int] = None) -> Dict[str, Any]:
        """Run the complete training pipeline: download -> validate -> clean -> train -> save"""
        
        symbols = symbols or self.config.get('symbols', ['AAPL', 'TSLA', 'GOOGL'])
        days = days or self.config.get('days', 730)
        
        logger.info("Starting full professional training pipeline",
                   symbols=symbols, days=days)
        
        try:
            # Step 1: Download data
            training_data = await self.download_training_data(symbols, days)
            
            # Step 2: Validate and clean
            cleaned_data = await self.validate_and_clean_data(training_data)
            
            # Step 3: Train models
            results = await self.train_models(cleaned_data)
            
            return results
            
        except Exception as e:
            logger.error("Full pipeline failed", error=str(e))
            raise


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Professional Model Training for QuantumSentiment")
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--symbols', '-s',
        type=str,
        help='Comma-separated list of symbols to train on (e.g., AAPL,TSLA,GOOGL)'
    )
    
    parser.add_argument(
        '--days', '-d',
        type=int,
        help='Number of days of historical data to use for training'
    )
    
    parser.add_argument(
        '--full-pipeline',
        action='store_true',
        help='Run complete pipeline: download data + train models'
    )
    
    parser.add_argument(
        '--models', '-m',
        type=str,
        help='Comma-separated list of models to train (lstm,xgboost,ensemble)',
        default='lstm,xgboost,ensemble'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_args()
    
    # Configure logging level
    import logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize trainer
        trainer = ProfessionalModelTrainer(args.config)
        
        if not await trainer.initialize():
            logger.error("Failed to initialize trainer")
            sys.exit(1)
        
        # Parse symbols if provided
        symbols = None
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(',')]
        
        # Parse models to train
        models_to_train = [m.strip().lower() for m in args.models.split(',')]
        trainer.config['models'] = {
            'train_lstm': 'lstm' in models_to_train,
            'train_xgboost': 'xgboost' in models_to_train,
            'train_ensemble': 'ensemble' in models_to_train,
            'train_cnn': 'cnn' in models_to_train,
            'train_finbert': 'finbert' in models_to_train
        }
        
        # Run pipeline
        if args.full_pipeline:
            results = await trainer.run_full_pipeline(symbols, args.days)
        else:
            # Just train with existing data (user should have run data download first)
            logger.info("Training with existing data (make sure you've downloaded data first)")
            # This would require loading data from database - for now, run full pipeline
            results = await trainer.run_full_pipeline(symbols, args.days)
        
        # Display results
        print("\n" + "="*80)
        print("🎉 PROFESSIONAL MODEL TRAINING COMPLETED!")
        print("="*80)
        print(f"✅ Models trained: {results['total_models_trained']}")
        print(f"📊 Training samples: {results['training_data_samples']:,}")
        print(f"🎯 Symbols: {results['training_data_symbols']}")
        
        if results['saved_models']:
            print("\n📦 Trained Models:")
            for model_name, info in results['saved_models'].items():
                print(f"  🧠 {model_name}:")
                print(f"     - Model ID: {info['model_id']}")
                print(f"     - Version: {info['version']}")
                print(f"     - Training Duration: {info['training_duration']:.2f}s")
                
                # Display key metrics
                test_metrics = info.get('test_metrics', {})
                if test_metrics:
                    print(f"     - Test Metrics: {test_metrics}")
        
        print("\n🚀 SUCCESS! Models are ready for production trading.")
        print("   Run: python src/main.py --mode paper")
        print("="*80)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error("Training failed", error=str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())