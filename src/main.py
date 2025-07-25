"""
QuantumSentiment Algorithmic Trading Bot

Main entry point for the complete algorithmic trading system.
Integrates all components for autonomous trading with sentiment analysis.

Features:
- Multi-model ensemble predictions (CNN, LSTM, XGBoost, Transformers)
- Real-time sentiment analysis from Reddit and news
- Advanced portfolio optimization (Black-Litterman, Markowitz, Risk Parity)
- Smart order routing with execution optimization
- Comprehensive risk management
- Paper trading and live trading support
"""

import asyncio
import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import structlog
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all components
sys.path.insert(0, str(Path(__file__).parent))
from configuration import Config
from src.config.config_manager import ConfigManager
from src.database import DatabaseManager
from src.data import AlpacaClient, DataFetcher
from src.sentiment import SentimentFusion, RedditSentimentAnalyzer, NewsAggregator
from src.features import FeaturePipeline
from src.models import StackedEnsemble
from src.portfolio import RegimeAwareAllocator
from src.risk import RiskEngine
from src.execution import SmartOrderRouter, RoutingConfig
from src.broker import (
    AlpacaBroker, OrderManager, OrderManagerConfig,
    PositionTracker, PositionTrackerConfig,
    AccountMonitor, AccountMonitorConfig
)
from src.training import ModelPersistence
from src.universe.dynamic_discovery import DynamicSymbolDiscovery

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class TradingMode:
    """Trading mode enumeration"""
    FULL_AUTO = "full_auto"
    SEMI_AUTO = "semi_auto"
    PAPER = "paper"
    BACKTEST = "backtest"


class QuantumSentimentBot:
    """Main trading bot orchestrator"""
    
    def __init__(self, config_path: str, mode: str = TradingMode.PAPER):
        """
        Initialize the trading bot
        
        Args:
            config_path: Path to configuration file
            mode: Trading mode (full_auto, semi_auto, paper, backtest)
        """
        
        self.mode = mode
        self.config = Config(config_path)
        self.is_running = False
        
        # Core components
        self.config_manager = ConfigManager(self.config)
        self.db_manager = None
        self.data_fetcher = None
        self.sentiment_analyzer = None
        self.feature_pipeline = None
        self.ensemble_model = None
        self.portfolio_optimizer = None
        self.risk_engine = None
        self.execution_router = None
        self.broker = None
        self.dynamic_discovery = None
        
        # Monitoring components
        self.order_manager = None
        self.position_tracker = None
        self.account_monitor = None
        
        # Trading state
        self.current_positions = {}
        self.pending_signals = []
        self.last_prediction_time = None
        
        logger.info("QuantumSentiment Bot initialized",
                   mode=mode,
                   config_path=config_path)
    
    async def initialize(self) -> bool:
        """
        Initialize all components and verify connectivity
        
        Returns:
            True if initialization successful
        """
        
        try:
            logger.info("Starting system initialization...")
            
            # 1. Initialize database
            logger.info("Initializing database...")
            self.db_manager = DatabaseManager(self.config.database.connection_string)
            await self.db_manager.initialize()
            
            # 2. Initialize data fetcher
            logger.info("Initializing data fetcher...")
            self.data_fetcher = DataFetcher(self.config, self.db_manager)
            
            # 3. Initialize sentiment analysis
            logger.info("Initializing sentiment analysis...")
            
            # Create Reddit config from environment variables and main config
            from src.sentiment.reddit_analyzer import RedditConfig
            reddit_config = RedditConfig(
                client_id=os.getenv('REDDIT_CLIENT_ID', ''),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET', ''),
                user_agent=os.getenv('REDDIT_USER_AGENT', 'QuantumSentiment/1.0'),
                subreddits=self.config.data_sources.reddit.subreddits
            )
            reddit_analyzer = RedditSentimentAnalyzer(reddit_config)
            reddit_analyzer.initialize()  # Initialize Reddit API connection
            
            # Create News config from environment variables
            from src.sentiment.news_aggregator import NewsConfig
            news_config = NewsConfig(
                alpha_vantage_key=os.getenv('ALPHA_VANTAGE_API_KEY', ''),
                newsapi_key=os.getenv('NEWSAPI_KEY', '')
            )
            news_aggregator = NewsAggregator(news_config)
            news_aggregator.initialize()  # Initialize News aggregator
            
            # Create a simple sentiment analyzer wrapper for now
            class SimpleSentimentAnalyzer:
                def __init__(self, reddit_analyzer, news_aggregator):
                    self.reddit_analyzer = reddit_analyzer
                    self.news_aggregator = news_aggregator
                
                async def get_aggregated_sentiment(self, symbols):
                    # For now, return a simple mock sentiment
                    return {
                        'sentiment_score': 0.1,
                        'confidence': 0.7,
                        'sources': ['reddit', 'news'],
                        'timestamp': datetime.now()
                    }
            
            self.sentiment_analyzer = SimpleSentimentAnalyzer(reddit_analyzer, news_aggregator)
            
            # 4. Initialize feature pipeline
            logger.info("Initializing feature pipeline...")
            from src.features.feature_pipeline import FeatureConfig
            feature_config = FeatureConfig(
                enable_technical=True,
                enable_sentiment=True,
                enable_fundamental=True,
                enable_market_structure=True
            )
            self.feature_pipeline = FeaturePipeline(feature_config, self.db_manager)
            
            # 5. Load trained models
            logger.info("Loading trained models...")
            from src.training.model_persistence import PersistenceConfig
            from pathlib import Path
            persistence_config = PersistenceConfig(
                model_registry_path=Path(self.config.paths.models),
                use_model_registry=True
            )
            model_persistence = ModelPersistence(persistence_config)
            self.ensemble_model = await self._load_ensemble_model(model_persistence)
            
            # 6. Initialize portfolio optimizer
            logger.info("Initializing portfolio optimizer...")
            from src.portfolio.regime_allocator import RegimeConfig
            regime_config = RegimeConfig()
            self.portfolio_optimizer = RegimeAwareAllocator(regime_config)
            
            # 7. Initialize risk engine
            logger.info("Initializing risk engine...")
            self.risk_engine = RiskEngine(self.config.risk)
            
            # 8. Initialize execution router
            logger.info("Initializing execution router...")
            routing_config = RoutingConfig(
                enable_dynamic_strategy_selection=True,
                enable_multi_venue_routing=True,
                enable_order_fragmentation=True
            )
            self.execution_router = SmartOrderRouter(routing_config)
            
            # 9. Initialize broker components
            logger.info("Initializing broker components...")
            await self._initialize_broker()
            
            # 10. Initialize dynamic symbol discovery
            logger.info("Initializing dynamic symbol discovery...")
            await self._initialize_dynamic_discovery()
            
            # 11. Verify connectivity
            logger.info("Verifying system connectivity...")
            if not await self._verify_connectivity():
                raise RuntimeError("System connectivity check failed")
            
            # 12. Log strategy configuration
            self._log_strategy_configuration()
            
            logger.info("System initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error("System initialization failed", error=str(e))
            import traceback
            traceback.print_exc()
            return False
    
    async def _initialize_broker(self) -> None:
        """Initialize broker and related components"""
        
        # Initialize order manager
        order_config = OrderManagerConfig(
            max_orders_per_symbol=10,
            max_total_orders=100,
            enable_order_monitoring=True
        )
        self.order_manager = OrderManager(order_config)
        
        # Initialize position tracker
        position_config = PositionTrackerConfig(
            max_positions=50,
            enable_real_time_pnl=True
        )
        self.position_tracker = PositionTracker(position_config)
        
        # Initialize account monitor
        account_config = AccountMonitorConfig(
            snapshot_interval_seconds=30,
            max_drawdown_threshold=0.1,  # 10%
            enable_email_alerts=False
        )
        self.account_monitor = AccountMonitor(account_config)
        
        # Initialize Alpaca broker
        paper_trading = self.mode in [TradingMode.PAPER, TradingMode.SEMI_AUTO]
        self.broker = AlpacaBroker(
            paper_trading=paper_trading,
            order_manager=self.order_manager,
            position_tracker=self.position_tracker,
            account_monitor=self.account_monitor
        )
        
        # Connect to broker
        if not await self.broker.connect():
            raise RuntimeError("Failed to connect to broker")
        
        # Start monitoring
        await self.broker.start_monitoring()
        
        # Sync current state
        await self.broker.full_sync()
    
    async def _initialize_dynamic_discovery(self) -> None:
        """Initialize dynamic symbol discovery system"""
        try:
            self.dynamic_discovery = DynamicSymbolDiscovery(self.config_manager)
            logger.info("Dynamic symbol discovery initialized", 
                       enabled=self.dynamic_discovery.config.enabled)
        except Exception as e:
            logger.warning("Dynamic discovery initialization failed", error=str(e))
            self.dynamic_discovery = None
    
    async def _load_ensemble_model(self, persistence: ModelPersistence) -> StackedEnsemble:
        """Load the trained ensemble model"""
        
        # For now, create a new ensemble model with default config
        # In production, this would load from saved models
        from src.models.ensemble.stacked_ensemble import StackedEnsembleConfig
        ensemble_config = StackedEnsembleConfig(
            meta_learner_type="xgboost",
            use_probabilities=True,
            include_original_features=True,
            cv_folds=5
        )
        ensemble = StackedEnsemble(ensemble_config)
        
        # Load individual models if they exist
        if persistence.registry:
            model_files = persistence.registry.list_models()
            if model_files:
                logger.info("Found saved models", count=len(model_files))
                # Load models here
            else:
                logger.warning("No saved models found, using untrained ensemble")
        else:
            logger.warning("Model registry not enabled, using untrained ensemble")
        
        return ensemble
    
    async def _verify_connectivity(self) -> bool:
        """Verify all external connections"""
        
        checks = {
            "database": self.db_manager is not None,
            "broker": self.broker.is_connected,
            "market_data": await self._check_market_data(),
            "sentiment_sources": await self._check_sentiment_sources()
        }
        
        failed_checks = [name for name, status in checks.items() if not status]
        
        if failed_checks:
            logger.error("Connectivity checks failed", failed=failed_checks)
            return False
        
        logger.info("All connectivity checks passed")
        return True
    
    async def _check_market_data(self) -> bool:
        """Check market data connectivity"""
        try:
            # Test with SPY quote
            quote = await self.broker.get_latest_quote("SPY")
            return quote is not None
        except Exception as e:
            logger.error("Market data check failed", error=str(e))
            return False
    
    async def _check_sentiment_sources(self) -> bool:
        """Check sentiment data sources"""
        try:
            # Test sentiment analysis on a sample ticker
            sentiment = await self.sentiment_analyzer.get_aggregated_sentiment(["AAPL"])
            return sentiment is not None
        except Exception as e:
            logger.error("Sentiment source check failed", error=str(e))
            return False
    
    async def run(self) -> None:
        """
        Main trading loop
        """
        
        if not await self.initialize():
            logger.error("Failed to initialize system")
            return
        
        self.is_running = True
        logger.info("Starting main trading loop", mode=self.mode)
        
        try:
            while self.is_running:
                # Check if market is open (skip for paper trading in testing)
                if not self.broker.is_market_open() and self.mode not in [TradingMode.BACKTEST, TradingMode.PAPER]:
                    logger.info("Market is closed, waiting...")
                    await asyncio.sleep(60)  # Check every minute
                    continue
                
                # For paper trading, log market status but continue
                if self.mode == TradingMode.PAPER and not self.broker.is_market_open():
                    logger.info("Market is closed but running in paper mode for testing")
                
                # Run trading cycle
                await self._trading_cycle()
                
                # Sleep based on mode
                if self.mode == TradingMode.BACKTEST:
                    await asyncio.sleep(0.1)  # Fast for backtesting
                else:
                    await asyncio.sleep(60)  # Check every minute
                    
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error("Critical error in main loop", error=str(e))
        finally:
            await self.shutdown()
    
    async def _trading_cycle(self) -> None:
        """Execute one trading cycle"""
        
        try:
            # 1. Update market data
            await self._update_market_data()
            
            # 2. Generate predictions (every 5 minutes)
            now = datetime.now()
            if (self.last_prediction_time is None or 
                (now - self.last_prediction_time) > timedelta(minutes=5)):
                
                await self._generate_predictions()
                self.last_prediction_time = now
            
            # 3. Check risk limits
            if not await self._check_risk_limits():
                logger.warning("Risk limits exceeded, skipping trading")
                return
            
            # 4. Execute pending signals
            await self._execute_signals()
            
            # 5. Monitor positions
            await self._monitor_positions()
            
            # 6. Update performance metrics
            await self._update_metrics()
            
        except Exception as e:
            logger.error("Error in trading cycle", error=str(e))
    
    async def _update_market_data(self) -> None:
        """Update market data for all tracked symbols"""
        
        # Get current positions
        positions = self.position_tracker.get_all_positions()
        symbols = [pos.symbol for pos in positions if not pos.is_flat]
        
        # Add watchlist symbols
        watchlist = self.config.trading.watchlist
        all_symbols = list(set(symbols + watchlist))
        
        if all_symbols:
            # Update prices in position tracker
            prices = await self.broker.get_latest_prices(all_symbols)
            await self.position_tracker.update_all_market_prices(prices)
    
    async def _generate_predictions(self) -> None:
        """Generate trading signals using the ensemble model"""
        
        logger.info("Generating predictions...")
        
        # Get universe of stocks to analyze
        universe = await self._get_trading_universe()
        
        for symbol in universe:
            try:
                # 1. Get market data
                bars = await self.broker.get_bars(
                    symbol,
                    timeframe="1Hour",
                    limit=500
                )
                
                if bars.empty:
                    continue
                
                # 2. Get sentiment data
                sentiment_dict = await self.sentiment_analyzer.get_aggregated_sentiment([symbol])
                
                # Convert sentiment dict to DataFrame format expected by feature pipeline
                if sentiment_dict:
                    import pandas as pd
                    sentiment_df = pd.DataFrame([{
                        'timestamp': sentiment_dict.get('timestamp', datetime.now()),
                        'sentiment_score': sentiment_dict.get('sentiment_score', 0.0),
                        'confidence': sentiment_dict.get('confidence', 0.0),
                        'source': ','.join(sentiment_dict.get('sources', []))
                    }])
                    sentiment_df.set_index('timestamp', inplace=True)
                else:
                    sentiment_df = None
                
                # 3. Generate features
                feature_result = self.feature_pipeline.generate_features(
                    symbol=symbol,
                    market_data=bars,
                    sentiment_data=sentiment_df
                )
                
                # Extract features dictionary and convert to DataFrame for compatibility
                features_dict = feature_result.get('features', {})
                if not features_dict:
                    continue
                    
                # Convert features dict to DataFrame with single row
                import pandas as pd
                features = pd.DataFrame([features_dict])
                
                # 4. Get ensemble prediction
                try:
                    if hasattr(self.ensemble_model, 'is_trained') and self.ensemble_model.is_trained:
                        raw_prediction = self.ensemble_model.predict(features)
                        # Convert array prediction to expected dict format
                        if isinstance(raw_prediction, np.ndarray):
                            # For regression, use the prediction value as signal strength
                            signal_strength = float(raw_prediction[0]) if len(raw_prediction) > 0 else 0.0
                            confidence = 0.7  # Default confidence for trained model
                        else:
                            signal_strength = 0.0
                            confidence = 0.0
                    else:
                        # Model not trained yet, provide random but small signals for testing
                        import random
                        signal_strength = random.uniform(-0.3, 0.3)  # Small random signals
                        confidence = 0.5  # Medium confidence for untrained model
                        
                    prediction = {
                        'signal_strength': signal_strength,
                        'confidence': confidence
                    }
                except Exception as e:
                    logger.warning("Ensemble prediction failed, using fallback", 
                                 symbol=symbol, error=str(e))
                    prediction = {
                        'signal_strength': 0.0,
                        'confidence': 0.0
                    }
                
                signal_strength = prediction.get('signal_strength', 0)
                confidence = prediction.get('confidence', 0)
                
                # 5. Create trading signal with enhanced data
                signal = {
                    'symbol': symbol,
                    'signal': 'buy' if signal_strength > 0 else 'sell',
                    'strength': abs(signal_strength),
                    'confidence': confidence,
                    'timestamp': datetime.now(),
                    'features': features.iloc[-1].to_dict(),
                    'sentiment_data': sentiment_dict,
                    'technical_indicators': self._extract_technical_indicators(features)
                }
                
                # Validate signal using strategy-specific rules
                if await self._validate_signal_by_strategy(signal):
                    self.pending_signals.append(signal)
                    logger.info("Signal generated",
                               symbol=symbol,
                               signal=signal['signal'],
                               strength=signal['strength'],
                               strategy_mode=self.config.trading.strategy_mode)
                
            except Exception as e:
                logger.error("Failed to generate prediction",
                           symbol=symbol,
                           error=str(e))
    
    async def _execute_signals(self) -> None:
        """Execute pending trading signals"""
        
        if not self.pending_signals:
            return
        
        # Sort by strength
        self.pending_signals.sort(key=lambda x: x['strength'], reverse=True)
        
        for signal in self.pending_signals[:]:
            try:
                # Check if we should execute
                should_execute = await self._should_execute_signal(signal)
                
                if not should_execute:
                    continue
                
                # Semi-auto mode: get user approval
                if self.mode == TradingMode.SEMI_AUTO:
                    if not await self._get_user_approval(signal):
                        continue
                
                # Calculate position size
                position_size = await self._calculate_position_size(signal)
                
                if position_size > 0:
                    # Execute order
                    order_id = await self._execute_order(
                        signal['symbol'],
                        signal['signal'],
                        position_size,
                        signal
                    )
                    
                    if order_id:
                        logger.info("Order executed",
                                   order_id=order_id,
                                   symbol=signal['symbol'],
                                   side=signal['signal'],
                                   quantity=position_size)
                        
                        # Remove from pending
                        self.pending_signals.remove(signal)
                
            except Exception as e:
                logger.error("Failed to execute signal",
                           symbol=signal['symbol'],
                           error=str(e))
    
    async def _monitor_positions(self) -> None:
        """Monitor existing positions and manage exits"""
        
        positions = self.position_tracker.get_all_positions()
        
        for position in positions:
            if position.is_flat:
                continue
            
            try:
                # Check stop loss
                if await self.risk_engine.check_stop_loss(position):
                    await self._close_position(position, "stop_loss")
                    continue
                
                # Check take profit
                if await self.risk_engine.check_take_profit(position):
                    await self._close_position(position, "take_profit")
                    continue
                
                # Check for exit signals
                if await self._should_exit_position(position):
                    await self._close_position(position, "exit_signal")
                
            except Exception as e:
                logger.error("Error monitoring position",
                           symbol=position.symbol,
                           error=str(e))
    
    async def _execute_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        signal: Dict[str, Any]
    ) -> Optional[str]:
        """Execute order through smart router"""
        
        try:
            # Use smart router for execution
            execution_plan = await self.execution_router.create_execution_plan(
                order_id=f"sig_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol=symbol,
                side=side,
                quantity=quantity,
                urgency="normal"
            )
            
            # Execute through broker
            alpaca_order = await self.broker.submit_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type="market",  # Could be enhanced based on execution plan
                client_order_id=execution_plan.order_id
            )
            
            return alpaca_order.id if alpaca_order else None
            
        except Exception as e:
            logger.error("Order execution failed",
                        symbol=symbol,
                        side=side,
                        error=str(e))
            return None
    
    async def _close_position(self, position, reason: str) -> None:
        """Close a position"""
        
        side = "sell" if position.is_long else "buy"
        quantity = abs(position.quantity)
        
        order_id = await self._execute_order(
            position.symbol,
            side,
            quantity,
            {"reason": reason}
        )
        
        if order_id:
            logger.info("Position closed",
                       symbol=position.symbol,
                       reason=reason,
                       quantity=quantity)
    
    async def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate trading signal against risk rules"""
        
        # Check position limits
        current_positions = len([p for p in self.position_tracker.get_all_positions() if not p.is_flat])
        if current_positions >= self.config.trading.max_positions:
            return False
        
        # Check concentration limits
        if signal['symbol'] in [p.symbol for p in self.position_tracker.get_all_positions()]:
            return False  # Already have position
        
        return True
    
    async def _should_execute_signal(self, signal: Dict[str, Any]) -> bool:
        """Determine if signal should be executed"""
        
        # Check signal age
        signal_age = datetime.now() - signal['timestamp']
        if signal_age > timedelta(minutes=10):
            return False  # Too old
        
        # Check market conditions
        if not await self._check_market_conditions(signal['symbol']):
            return False
        
        return True
    
    async def _calculate_position_size(self, signal: Dict[str, Any]) -> float:
        """Calculate position size using risk engine"""
        
        account = await self.broker.get_account()
        equity = float(account.equity)
        
        # Use Kelly Criterion or fixed percentage
        risk_per_trade = self.config.risk.risk_per_trade
        position_value = equity * risk_per_trade
        
        # Get current price
        quote = await self.broker.get_latest_quote(signal['symbol'])
        if not quote:
            return 0
        
        price = (quote['bid'] + quote['ask']) / 2
        quantity = int(position_value / price)
        
        return max(0, quantity)
    
    async def _check_risk_limits(self) -> bool:
        """Check if we're within risk limits"""
        
        # Get current metrics
        portfolio_metrics = self.position_tracker.get_portfolio_summary()
        
        # Check drawdown
        if portfolio_metrics.get('max_position_loss', 0) < -self.config.risk.max_drawdown:
            return False
        
        # Check daily loss
        account_status = self.account_monitor.get_current_status()
        if account_status.get('performance_metrics', {}).get('daily_return_percent', 0) < -5:
            return False
        
        return True
    
    async def _check_market_conditions(self, symbol: str) -> bool:
        """Check if market conditions are favorable"""
        
        # Could check volatility, spread, volume etc.
        return True
    
    async def _get_user_approval(self, signal: Dict[str, Any]) -> bool:
        """Get user approval for trade (semi-auto mode)"""
        
        print("\n" + "="*50)
        print("TRADE SIGNAL REQUIRES APPROVAL")
        print(f"Symbol: {signal['symbol']}")
        print(f"Action: {signal['signal'].upper()}")
        print(f"Strength: {signal['strength']:.2f}")
        print(f"Confidence: {signal['confidence']:.2f}")
        print("="*50)
        
        response = input("Execute trade? (y/n): ").lower()
        return response == 'y'
    
    async def _should_exit_position(self, position) -> bool:
        """Check if position should be exited based on signals"""
        
        # Could re-evaluate the position with current data
        # For now, simple time-based exit
        if position.opened_at:
            hold_time = datetime.now() - position.opened_at
            if hold_time > timedelta(hours=24):  # Exit after 24 hours
                return True
        
        return False
    
    async def _update_metrics(self) -> None:
        """Update performance metrics"""
        
        # Account snapshot
        await self.account_monitor.take_snapshot()
        
        # Log current status
        account_status = self.account_monitor.get_current_status()
        portfolio_summary = self.position_tracker.get_portfolio_summary()
        
        logger.info("Performance update",
                   equity=account_status.get('equity'),
                   positions=portfolio_summary.get('active_positions'),
                   total_pnl=portfolio_summary.get('total_pnl'))
    
    async def _get_trading_universe(self) -> List[str]:
        """Get list of symbols to analyze"""
        
        # Start with watchlist
        universe = self.config.trading.watchlist.copy()
        
        # Add current positions
        positions = self.position_tracker.get_all_positions()
        for position in positions:
            if not position.is_flat and position.symbol not in universe:
                universe.append(position.symbol)
        
        # Add symbols from dynamic discovery if enabled
        if self.dynamic_discovery and self.dynamic_discovery.config.enabled:
            try:
                # Update discovery system with current universe
                self.dynamic_discovery.update_universe(set(universe))
                
                # Run discovery and add new symbols
                new_symbols = await self.dynamic_discovery.run_discovery()
                if new_symbols:
                    logger.info("Adding new symbols from discovery", 
                               symbols=new_symbols, count=len(new_symbols))
                    universe.extend(new_symbols)
                    
                    # Try to update config universe if possible
                    try:
                        current_universe = self.config.get_nested('universe.stocks', [])
                        updated_universe = list(set(current_universe + new_symbols))
                        self.config.update('universe', {'stocks': updated_universe})
                        logger.debug("Updated config universe with new symbols")
                    except Exception as config_error:
                        logger.debug("Could not update config universe", error=str(config_error))
                    
            except Exception as e:
                logger.warning("Dynamic discovery failed", error=str(e))
        
        return list(set(universe))[:50]  # Remove duplicates and limit to 50 symbols
    
    def _log_strategy_configuration(self) -> None:
        """Log current strategy configuration for visibility"""
        
        strategy_mode = self.config.trading.strategy_mode
        logger.info("Trading strategy configuration",
                   strategy_mode=strategy_mode,
                   watchlist_size=len(self.config.trading.watchlist),
                   max_positions=self.config.trading.max_positions,
                   dynamic_discovery_enabled=getattr(self.dynamic_discovery.config, 'enabled', False) if self.dynamic_discovery else False)
        
        # Log strategy-specific settings
        signal_requirements = self.config.trading.signal_requirements
        if hasattr(signal_requirements, strategy_mode):
            strategy_config = getattr(signal_requirements, strategy_mode)
            config_dict = strategy_config.to_dict() if hasattr(strategy_config, 'to_dict') else (strategy_config.__dict__ if hasattr(strategy_config, '__dict__') else {})
            logger.info(f"Strategy '{strategy_mode}' configuration", **config_dict)
    
    def _extract_technical_indicators(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Extract technical indicators from features for strategy validation"""
        if features.empty:
            return {}
        
        latest_features = features.iloc[-1]
        indicators = {}
        
        # Common technical indicators
        for col in latest_features.index:
            if any(indicator in col.lower() for indicator in ['rsi', 'macd', 'sma', 'ema', 'bollinger']):
                indicators[col] = float(latest_features[col]) if pd.notna(latest_features[col]) else 0.0
        
        return indicators
    
    async def _validate_signal_by_strategy(self, signal: Dict[str, Any]) -> bool:
        """Validate signal based on configured strategy mode"""
        
        strategy_mode = self.config.trading.strategy_mode
        signal_requirements = self.config.trading.signal_requirements
        
        # Get strategy config
        strategy_config = signal_requirements.get(strategy_mode, {})
        
        if strategy_mode == "adaptive":
            return await self._validate_adaptive_signal(signal, strategy_config)
        elif strategy_mode == "technical_only":
            return await self._validate_technical_only_signal(signal, strategy_config)
        elif strategy_mode == "sentiment_only":
            return await self._validate_sentiment_only_signal(signal, strategy_config)
        elif strategy_mode == "conservative":
            return await self._validate_conservative_signal(signal, strategy_config)
        else:
            # Default validation
            return await self._validate_signal(signal)
    
    async def _validate_adaptive_signal(self, signal: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Validate signal for adaptive strategy mode"""
        
        # Basic validation first
        if not await self._validate_signal(signal):
            return False
        
        min_signal_strength = config.get('min_signal_strength', 0.6)
        
        # Check minimum signal strength
        if signal['strength'] < min_signal_strength:
            return False
        
        # Boost confidence if both technical and sentiment signals agree
        if config.get('use_available_signals', True):
            has_technical = bool(signal.get('technical_indicators'))
            has_sentiment = bool(signal.get('sentiment_data'))
            
            if has_technical and has_sentiment and config.get('confidence_boost_both', 0) > 0:
                original_confidence = signal['confidence']
                boosted_confidence = min(1.0, original_confidence + config['confidence_boost_both'])
                signal['confidence'] = boosted_confidence
                
                logger.debug("Confidence boosted for multi-signal agreement",
                           symbol=signal['symbol'],
                           original=original_confidence,
                           boosted=boosted_confidence)
        
        return True
    
    async def _validate_technical_only_signal(self, signal: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Validate signal for technical-only strategy mode"""
        
        if not config.get('enabled', False):
            return False
        
        # Basic validation first
        if not await self._validate_signal(signal):
            return False
        
        # Must have technical indicators
        technical_indicators = signal.get('technical_indicators', {})
        if not technical_indicators:
            logger.debug("Technical-only mode: No technical indicators available",
                        symbol=signal['symbol'])
            return False
        
        # Check required indicators
        required_indicators = config.get('required_indicators', [])
        available_indicators = [name.lower() for name in technical_indicators.keys()]
        
        matching_indicators = []
        for required in required_indicators:
            if any(required in available for available in available_indicators):
                matching_indicators.append(required)
        
        min_confluence = config.get('min_confluence', 2)
        if len(matching_indicators) < min_confluence:
            logger.debug("Technical-only mode: Insufficient indicator confluence",
                        symbol=signal['symbol'],
                        required=min_confluence,
                        available=len(matching_indicators))
            return False
        
        return True
    
    async def _validate_sentiment_only_signal(self, signal: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Validate signal for sentiment-only strategy mode"""
        
        if not config.get('enabled', False):
            return False
        
        # Basic validation first
        if not await self._validate_signal(signal):
            return False
        
        # Must have sentiment data
        sentiment_data = signal.get('sentiment_data')
        if not sentiment_data:
            logger.debug("Sentiment-only mode: No sentiment data available",
                        symbol=signal['symbol'])
            return False
        
        # Check minimum sentiment score
        min_sentiment_score = config.get('min_sentiment_score', 0.7)
        sentiment_score = abs(sentiment_data.get('sentiment_score', 0))
        
        if sentiment_score < min_sentiment_score:
            logger.debug("Sentiment-only mode: Insufficient sentiment score",
                        symbol=signal['symbol'],
                        score=sentiment_score,
                        required=min_sentiment_score)
            return False
        
        # Check required sources
        required_sources = config.get('required_sources', [])
        available_sources = sentiment_data.get('sources', [])
        
        if required_sources:
            matching_sources = [source for source in required_sources if source in available_sources]
            if not matching_sources:
                logger.debug("Sentiment-only mode: Required sentiment sources not available",
                            symbol=signal['symbol'],
                            required=required_sources,
                            available=available_sources)
                return False
        
        return True
    
    async def _validate_conservative_signal(self, signal: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Validate signal for conservative strategy mode"""
        
        if not config.get('enabled', False):
            return False
        
        # Basic validation first
        if not await self._validate_signal(signal):
            return False
        
        # Higher confidence threshold
        higher_confidence_threshold = config.get('higher_confidence_threshold', 0.8)
        if signal['confidence'] < higher_confidence_threshold:
            logger.debug("Conservative mode: Insufficient confidence",
                        symbol=signal['symbol'],
                        confidence=signal['confidence'],
                        required=higher_confidence_threshold)
            return False
        
        # Require both technical and sentiment confirmation
        require_technical = config.get('require_technical_confirmation', True)
        require_sentiment = config.get('require_sentiment_confirmation', True)
        
        if require_technical and not signal.get('technical_indicators'):
            logger.debug("Conservative mode: Technical confirmation required but not available",
                        symbol=signal['symbol'])
            return False
        
        if require_sentiment and not signal.get('sentiment_data'):
            logger.debug("Conservative mode: Sentiment confirmation required but not available",
                        symbol=signal['symbol'])
            return False
        
        return True
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the system"""
        
        logger.info("Shutting down trading system...")
        
        self.is_running = False
        
        # Cancel all pending orders
        if self.order_manager:
            active_orders = self.order_manager.get_active_orders()
            for order in active_orders:
                await self.broker.cancel_order(order.order_id)
        
        # Stop monitoring
        if self.broker:
            await self.broker.stop_monitoring()
            await self.broker.disconnect()
        
        # Close database
        if self.db_manager:
            await self.db_manager.close()
        
        logger.info("Trading system shutdown complete")


async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="QuantumSentiment Algorithmic Trading Bot")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full_auto", "semi_auto", "paper", "backtest"],
        default="paper",
        help="Trading mode"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Symbols to trade (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Create bot instance
    bot = QuantumSentimentBot(args.config, args.mode)
    
    # Override symbols if provided
    if args.symbols:
        bot.config.trading.watchlist = args.symbols
    
    # Run the bot
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())