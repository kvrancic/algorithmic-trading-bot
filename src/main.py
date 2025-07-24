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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all components
from src.config import Config
from src.database import DatabaseManager
from src.data import AlpacaClient, DataFetcher
from src.sentiment import SentimentFusion, RedditAnalyzer, NewsAggregator
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
        self.db_manager = None
        self.data_fetcher = None
        self.sentiment_analyzer = None
        self.feature_pipeline = None
        self.ensemble_model = None
        self.portfolio_optimizer = None
        self.risk_engine = None
        self.execution_router = None
        self.broker = None
        
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
            reddit_analyzer = RedditAnalyzer(self.config)
            news_aggregator = NewsAggregator(self.config)
            self.sentiment_analyzer = SentimentFusion()
            self.sentiment_analyzer.add_analyzer("reddit", reddit_analyzer)
            self.sentiment_analyzer.add_analyzer("news", news_aggregator)
            
            # 4. Initialize feature pipeline
            logger.info("Initializing feature pipeline...")
            self.feature_pipeline = FeaturePipeline(self.config)
            
            # 5. Load trained models
            logger.info("Loading trained models...")
            model_persistence = ModelPersistence(self.config.paths.models)
            self.ensemble_model = await self._load_ensemble_model(model_persistence)
            
            # 6. Initialize portfolio optimizer
            logger.info("Initializing portfolio optimizer...")
            self.portfolio_optimizer = RegimeAwareAllocator(self.config)
            
            # 7. Initialize risk engine
            logger.info("Initializing risk engine...")
            self.risk_engine = RiskEngine(self.config.risk)
            
            # 8. Initialize execution router
            logger.info("Initializing execution router...")
            routing_config = RoutingConfig(
                enable_smart_routing=True,
                enable_multi_venue_routing=True,
                enable_timing_optimization=True
            )
            self.execution_router = SmartOrderRouter(routing_config)
            
            # 9. Initialize broker components
            logger.info("Initializing broker components...")
            await self._initialize_broker()
            
            # 10. Verify connectivity
            logger.info("Verifying system connectivity...")
            if not await self._verify_connectivity():
                raise RuntimeError("System connectivity check failed")
            
            logger.info("System initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error("System initialization failed", error=str(e))
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
    
    async def _load_ensemble_model(self, persistence: ModelPersistence) -> StackedEnsemble:
        """Load the trained ensemble model"""
        
        # For now, create a new ensemble model
        # In production, this would load from saved models
        ensemble = StackedEnsemble()
        
        # Load individual models if they exist
        model_files = persistence.list_models()
        if model_files:
            logger.info("Found saved models", count=len(model_files))
            # Load models here
        else:
            logger.warning("No saved models found, using untrained ensemble")
        
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
                # Check if market is open
                if not self.broker.is_market_open() and self.mode != TradingMode.BACKTEST:
                    logger.info("Market is closed, waiting...")
                    await asyncio.sleep(60)  # Check every minute
                    continue
                
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
        universe = self._get_trading_universe()
        
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
                sentiment = await self.sentiment_analyzer.get_aggregated_sentiment([symbol])
                
                # 3. Generate features
                features = self.feature_pipeline.create_features(
                    price_data=bars,
                    sentiment_data=sentiment,
                    symbol=symbol
                )
                
                # 4. Get ensemble prediction
                prediction = await self.ensemble_model.predict(features)
                signal_strength = prediction.get('signal_strength', 0)
                confidence = prediction.get('confidence', 0)
                
                # 5. Create trading signal if strong enough
                if abs(signal_strength) > 0.7 and confidence > 0.6:
                    signal = {
                        'symbol': symbol,
                        'signal': 'buy' if signal_strength > 0 else 'sell',
                        'strength': abs(signal_strength),
                        'confidence': confidence,
                        'timestamp': datetime.now(),
                        'features': features.iloc[-1].to_dict()
                    }
                    
                    # Risk check
                    if await self._validate_signal(signal):
                        self.pending_signals.append(signal)
                        logger.info("Signal generated",
                                   symbol=symbol,
                                   signal=signal['signal'],
                                   strength=signal['strength'])
                
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
        if current_positions >= self.config.risk.max_positions:
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
    
    def _get_trading_universe(self) -> List[str]:
        """Get list of symbols to analyze"""
        
        # Start with watchlist
        universe = self.config.trading.watchlist.copy()
        
        # Add current positions
        positions = self.position_tracker.get_all_positions()
        for position in positions:
            if not position.is_flat and position.symbol not in universe:
                universe.append(position.symbol)
        
        # Could add trending stocks from sentiment analysis
        
        return universe[:50]  # Limit to 50 symbols
    
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