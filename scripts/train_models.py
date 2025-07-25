#!/usr/bin/env python3
"""
Model Training Script for QuantumSentiment Trading Bot

This script trains and saves all models in the trading system:
- Price prediction LSTM
- Chart pattern CNN
- Market regime XGBoost
- Sentiment analysis FinBERT
- Stacked ensemble

Usage:
    python scripts/train_models.py --help
    python scripts/train_models.py --models lstm,xgboost --days 365
    python scripts/train_models.py --all --parallel
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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from training.training_pipeline import ModelTrainingPipeline, TrainingConfig
from training.model_persistence import ModelPersistence, PersistenceConfig
from data.data_fetcher import DataFetcher, FetcherConfig
from data.data_interface import DataInterface
from database.database import DatabaseManager
from features.feature_pipeline import FeaturePipeline, FeatureConfig

logger = structlog.get_logger(__name__)


class ModelTrainer:
    """Main class for training and saving models"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model trainer"""
        self.config = config
        
        # Initialize components
        self.db_manager = None
        self.data_fetcher = None
        self.training_pipeline = None
        self.model_persistence = None
        
        logger.info("Model trainer initialized")
    
    async def initialize(self) -> bool:
        """Initialize all components"""
        try:
            # Initialize database
            logger.info("Initializing database...")
            db_url = self.config.get('database_url', 'sqlite:///quantum_sentiment.db')
            self.db_manager = DatabaseManager(db_url)
            
            # Initialize data fetcher
            logger.info("Initializing data fetcher...")
            data_interface = DataInterface()
            fetcher_config = FetcherConfig()
            self.data_fetcher = DataFetcher(fetcher_config, data_interface)
            
            # Initialize training pipeline
            logger.info("Initializing training pipeline...")
            training_config = TrainingConfig(
                train_start_date=self.config.get('train_start_date', '2022-01-01'),
                train_end_date=self.config.get('train_end_date', '2024-12-31'),
                parallel_training=self.config.get('parallel_training', True),
                max_workers=self.config.get('max_workers', 4),
                train_lstm=self.config.get('train_lstm', True),
                train_cnn=self.config.get('train_cnn', True),
                train_xgboost=self.config.get('train_xgboost', True),
                train_finbert=self.config.get('train_finbert', False),  # Heavy compute
                train_ensemble=self.config.get('train_ensemble', True),
                model_save_dir=Path(self.config.get('model_save_dir', 'models'))
            )
            self.training_pipeline = ModelTrainingPipeline(training_config)
            
            # Initialize model persistence
            logger.info("Initializing model persistence...")
            persistence_config = PersistenceConfig(
                model_registry_path=Path(self.config.get('model_registry_path', 'model_registry')),
                use_model_registry=True,
                auto_cleanup=True,
                max_versions_per_model=5
            )
            self.model_persistence = ModelPersistence(persistence_config)
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error("Initialization failed", error=str(e))
            return False
    
    async def load_training_data(self, symbols: List[str], days: int) -> pd.DataFrame:
        """Load and prepare training data"""
        logger.info("Loading training data", symbols=symbols, days=days)
        
        all_data = []
        
        for symbol in symbols:
            try:
                # Get market data for multiple timeframes
                for timeframe in ['1Day', '1Hour']:
                    logger.info(f"Fetching {timeframe} data for {symbol}")
                    
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
                        logger.info(f"Loaded {len(symbol_data)} records for {symbol} {timeframe}")
                    else:
                        logger.warning(f"No data found for {symbol} {timeframe}")
                        
            except Exception as e:
                logger.error(f"Failed to load data for {symbol}", error=str(e))
        
        if not all_data:
            raise ValueError("No training data could be loaded")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Ensure timestamp column exists and is datetime
        if 'timestamp' not in combined_data.columns:
            combined_data.reset_index(inplace=True)
            if 'index' in combined_data.columns:
                combined_data.rename(columns={'index': 'timestamp'}, inplace=True)
        
        # Convert timestamp to datetime if it's not already
        combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'])
        
        # Sort by timestamp
        combined_data.sort_values('timestamp', inplace=True)
        
        logger.info("Training data loaded successfully", 
                   total_records=len(combined_data),
                   symbols=len(symbols),
                   date_range=f"{combined_data['timestamp'].min()} to {combined_data['timestamp'].max()}")
        
        return combined_data
    
    async def train_all_models(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """Train all configured models"""
        logger.info("Starting model training")
        
        # Train models using the pipeline
        trained_models = self.training_pipeline.train_all_models(
            price_data=training_data,
            text_data=None  # TODO: Add sentiment data if available
        )
        
        # Save models with proper metadata
        saved_models = {}
        
        for model_name, model in trained_models.items():
            try:
                logger.info(f"Saving model: {model_name}")
                
                # Calculate training data hash for versioning
                training_data_hash = self._calculate_data_hash(training_data)
                
                # Get training results
                training_results = self.training_pipeline.training_results.get(model_name, {})
                validation_metrics = training_results.get('validation_metrics', {})
                test_metrics = training_results.get('test_metrics', {})
                
                # Save model with metadata
                metadata = self.model_persistence.save_model(
                    model=model,
                    model_name=model_name,
                    training_data_hash=training_data_hash,
                    training_samples=len(training_data),
                    training_duration=training_results.get('training_time', 0.0),
                    validation_metrics=validation_metrics,
                    test_metrics=test_metrics,
                    description=f"{model_name} trained on {len(training_data)} samples",
                    tags=[f"version_{datetime.now().strftime('%Y%m%d')}", "production"],
                    author="QuantumSentiment Training Pipeline"
                )
                
                saved_models[model_name] = {
                    'model_id': metadata.model_id,
                    'version': metadata.version,
                    'path': str(metadata.model_path),
                    'validation_metrics': validation_metrics,
                    'test_metrics': test_metrics
                }
                
                logger.info(f"Model {model_name} saved successfully", 
                           model_id=metadata.model_id,
                           version=metadata.version)
                
            except Exception as e:
                logger.error(f"Failed to save model {model_name}", error=str(e))
        
        # Get training summary
        training_summary = self.training_pipeline.get_training_summary()
        
        return {
            'saved_models': saved_models,
            'training_summary': training_summary,
            'total_models': len(saved_models)
        }
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of training data for versioning"""
        import hashlib
        
        # Create a simple hash based on data shape and sample values
        data_info = f"{len(data)}_{data.shape[1]}_{data['timestamp'].min()}_{data['timestamp'].max()}"
        
        # Add sample of actual data
        if len(data) > 0:
            sample_data = data.head(10).to_string()
            data_info += sample_data
        
        return hashlib.md5(data_info.encode()).hexdigest()
    
    async def list_saved_models(self) -> Dict[str, List[str]]:
        """List all saved models"""
        if self.model_persistence:
            return self.model_persistence.list_available_models()
        return {}
    
    async def cleanup_old_models(self, days_threshold: int = 30):
        """Clean up old model versions"""
        if self.model_persistence:
            self.model_persistence.cleanup_old_models(days_threshold)
            logger.info("Model cleanup completed")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train and save trading models")
    
    parser.add_argument(
        '--models', '-m',
        type=str,
        help='Comma-separated list of models to train (lstm,cnn,xgboost,finbert,ensemble,all)',
        default='lstm,xgboost,ensemble'
    )
    
    parser.add_argument(
        '--symbols', '-s',
        type=str,
        help='Comma-separated list of symbols to train on',
        default='AAPL,TSLA,GOOGL,MSFT,SPY'
    )
    
    parser.add_argument(
        '--days', '-d',
        type=int,
        help='Number of days of historical data to use for training',
        default=730  # 2 years
    )
    
    parser.add_argument(
        '--parallel', '-p',
        action='store_true',
        help='Train models in parallel'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        help='Number of parallel workers',
        default=4
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for saved models',
        default='model_registry'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to training configuration file'
    )
    
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean up old model versions after training'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all saved models and exit'
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
        # Parse models to train
        models_to_train = args.models.lower().split(',')
        if 'all' in models_to_train:
            models_to_train = ['lstm', 'cnn', 'xgboost', 'finbert', 'ensemble']
        
        # Parse symbols
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
        
        # Create training configuration
        config = {
            'train_lstm': 'lstm' in models_to_train,
            'train_cnn': 'cnn' in models_to_train,
            'train_xgboost': 'xgboost' in models_to_train,
            'train_finbert': 'finbert' in models_to_train,
            'train_ensemble': 'ensemble' in models_to_train,
            'parallel_training': args.parallel,
            'max_workers': args.workers,
            'model_registry_path': args.output_dir,
            'model_save_dir': f"{args.output_dir}/saved_models"
        }
        
        # Initialize trainer
        trainer = ModelTrainer(config)
        
        if not await trainer.initialize():
            logger.error("Failed to initialize trainer")
            sys.exit(1)
        
        # List models if requested
        if args.list_models:
            saved_models = await trainer.list_saved_models()
            
            print("\n" + "="*60)
            print("SAVED MODELS")
            print("="*60)
            
            if saved_models:
                for model_name, versions in saved_models.items():
                    print(f"\n{model_name}:")
                    for version in versions:
                        print(f"  - {version}")
            else:
                print("No saved models found.")
            
            print("="*60)
            return
        
        # Load training data
        logger.info("Loading training data", symbols=symbols, days=args.days)
        training_data = await trainer.load_training_data(symbols, args.days)
        
        if training_data.empty:
            logger.error("No training data available")
            sys.exit(1)
        
        # Train models
        logger.info("Starting model training", models=models_to_train)
        results = await trainer.train_all_models(training_data)
        
        # Clean up old models if requested
        if args.cleanup:
            await trainer.cleanup_old_models()
        
        # Print results
        print("\n" + "="*80)
        print("MODEL TRAINING COMPLETED")
        print("="*80)
        print(f"Total models trained: {results['total_models']}")
        print(f"Training data: {len(training_data):,} samples")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Date range: {training_data['timestamp'].min()} to {training_data['timestamp'].max()}")
        
        if results['saved_models']:
            print("\nSaved Models:")
            for model_name, info in results['saved_models'].items():
                print(f"  {model_name}:")
                print(f"    - Model ID: {info['model_id']}")
                print(f"    - Version: {info['version']}")
                print(f"    - Path: {info['path']}")
                
                # Print metrics if available
                test_metrics = info.get('test_metrics', {})
                if test_metrics:
                    print(f"    - Test Metrics: {test_metrics}")
        
        print("\n" + "="*80)
        print("Models are now ready for use in the trading system!")
        print("The main trading loop will automatically load the latest trained models.")
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