#!/usr/bin/env python3
"""
System Status Check Script

Displays current status of the trading bot including:
- Account information
- Active positions
- Recent orders
- Performance metrics
- System health
"""

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


class StatusChecker:
    """Check and display trading system status"""
    
    def __init__(self):
        self.broker = None
        self.position_tracker = None
        self.account_monitor = None
        
    async def initialize(self):
        """Initialize connections"""
        try:
            from src.broker import AlpacaBroker, PositionTracker, AccountMonitor
            from src.broker import PositionTrackerConfig, AccountMonitorConfig
            
            # Initialize components
            self.position_tracker = PositionTracker(PositionTrackerConfig())
            self.account_monitor = AccountMonitor(AccountMonitorConfig())
            
            self.broker = AlpacaBroker(
                paper_trading=True,
                position_tracker=self.position_tracker,
                account_monitor=self.account_monitor
            )
            
            # Connect to broker
            connected = await self.broker.connect()
            if not connected:
                print(f"{RED}Failed to connect to broker{RESET}")
                return False
            
            # Sync data
            await self.broker.full_sync()
            return True
            
        except Exception as e:
            print(f"{RED}Initialization error: {e}{RESET}")
            return False
    
    def format_currency(self, value: float) -> str:
        """Format currency values"""
        if value >= 0:
            return f"{GREEN}${value:,.2f}{RESET}"
        else:
            return f"{RED}${value:,.2f}{RESET}"
    
    def format_percent(self, value: float) -> str:
        """Format percentage values"""
        if value >= 0:
            return f"{GREEN}+{value:.2f}%{RESET}"
        else:
            return f"{RED}{value:.2f}%{RESET}"
    
    async def display_account_info(self):
        """Display account information"""
        print(f"\n{BLUE}=== ACCOUNT INFORMATION ==={RESET}")
        
        try:
            account = await self.broker.get_account()
            if account:
                print(f"Status: {GREEN}{account.status}{RESET}")
                print(f"Equity: {self.format_currency(float(account.equity))}")
                print(f"Buying Power: {self.format_currency(float(account.buying_power))}")
                print(f"Cash: {self.format_currency(float(account.cash))}")
                print(f"Portfolio Value: {self.format_currency(float(account.portfolio_value))}")
                
                # Calculate daily change if available
                if hasattr(account, 'equity') and hasattr(account, 'last_equity'):
                    daily_change = float(account.equity) - float(account.last_equity)
                    daily_pct = (daily_change / float(account.last_equity)) * 100 if float(account.last_equity) > 0 else 0
                    print(f"Daily P&L: {self.format_currency(daily_change)} ({self.format_percent(daily_pct)})")
            else:
                print(f"{RED}Unable to retrieve account information{RESET}")
                
        except Exception as e:
            print(f"{RED}Error getting account info: {e}{RESET}")
    
    async def display_positions(self):
        """Display current positions"""
        print(f"\n{BLUE}=== ACTIVE POSITIONS ==={RESET}")
        
        positions = self.position_tracker.get_all_positions()
        active_positions = [p for p in positions if not p.is_flat]
        
        if not active_positions:
            print("No active positions")
            return
        
        # Update with latest prices
        symbols = [p.symbol for p in active_positions]
        prices = await self.broker.get_latest_prices(symbols)
        await self.position_tracker.update_all_market_prices(prices)
        
        # Display positions
        total_value = 0
        total_pnl = 0
        
        for position in active_positions:
            current_price = prices.get(position.symbol, position.average_price)
            market_value = abs(position.quantity * current_price)
            total_value += market_value
            total_pnl += position.total_pnl
            
            print(f"\n{position.symbol}:")
            print(f"  Side: {position.side.value.upper()}")
            print(f"  Quantity: {position.quantity:.2f}")
            print(f"  Avg Price: ${position.average_price:.2f}")
            print(f"  Current: ${current_price:.2f}")
            print(f"  Market Value: ${market_value:,.2f}")
            print(f"  Unrealized P&L: {self.format_currency(position.unrealized_pnl)}")
            print(f"  Realized P&L: {self.format_currency(position.realized_pnl)}")
            print(f"  Total P&L: {self.format_currency(position.total_pnl)} ({self.format_percent(position.pnl_percent)})")
        
        print(f"\n{BLUE}PORTFOLIO SUMMARY:{RESET}")
        print(f"Total Positions: {len(active_positions)}")
        print(f"Total Value: ${total_value:,.2f}")
        print(f"Total P&L: {self.format_currency(total_pnl)}")
    
    async def display_recent_orders(self):
        """Display recent orders"""
        print(f"\n{BLUE}=== RECENT ORDERS ==={RESET}")
        
        try:
            orders = await self.broker.get_orders(status="all", limit=10)
            
            if not orders:
                print("No recent orders")
                return
            
            for order in orders[:5]:  # Show last 5 orders
                print(f"\n{order.symbol} - {order.side.upper()} {order.qty}")
                print(f"  Type: {order.order_type}")
                print(f"  Status: {order.status}")
                print(f"  Submitted: {order.submitted_at}")
                
                if hasattr(order, 'filled_qty') and float(order.filled_qty) > 0:
                    print(f"  Filled: {order.filled_qty} @ ${float(order.filled_avg_price):.2f}")
                
                if order.limit_price:
                    print(f"  Limit: ${float(order.limit_price):.2f}")
                    
        except Exception as e:
            print(f"{RED}Error getting orders: {e}{RESET}")
    
    async def display_performance_metrics(self):
        """Display performance metrics"""
        print(f"\n{BLUE}=== PERFORMANCE METRICS ==={RESET}")
        
        # Get account monitor status
        status = self.account_monitor.get_current_status()
        
        if status.get('status') == 'no_data':
            print("No performance data available yet")
            return
        
        perf = status.get('performance_metrics', {})
        risk = status.get('risk_metrics', {})
        
        # Performance metrics
        if perf:
            print(f"\nPerformance:")
            print(f"  Total Return: {self.format_percent(perf.get('total_return_percent', 0))}")
            print(f"  Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
            print(f"  Max Drawdown: {RED}{perf.get('max_drawdown_percent', 0):.2f}%{RESET}")
            print(f"  Volatility: {perf.get('volatility_percent', 0):.2f}%")
        
        # Risk metrics
        if risk:
            print(f"\nRisk Metrics:")
            print(f"  Margin Utilization: {risk.get('margin_utilization_percent', 0):.1f}%")
            print(f"  Buying Power Ratio: {risk.get('buying_power_ratio', 0):.1f}%")
            print(f"  Max Position Concentration: {risk.get('max_position_concentration_percent', 0):.1f}%")
            print(f"  Day Trades: {risk.get('day_trade_count', 0)}")
    
    async def display_system_health(self):
        """Display system health status"""
        print(f"\n{BLUE}=== SYSTEM HEALTH ==={RESET}")
        
        # Check broker connection
        print(f"Broker Connection: {GREEN}Connected{RESET}" if self.broker.is_connected else f"{RED}Disconnected{RESET}")
        
        # Check market status
        market_open = self.broker.is_market_open()
        print(f"Market Status: {GREEN}Open{RESET}" if market_open else f"{YELLOW}Closed{RESET}")
        
        # Check for alerts
        alerts = self.account_monitor.get_alerts(resolved=False)
        if alerts:
            print(f"\n{YELLOW}Active Alerts:{RESET}")
            for alert in alerts[:3]:  # Show first 3 alerts
                print(f"  - [{alert.level.value.upper()}] {alert.message}")
        else:
            print(f"Alerts: {GREEN}None{RESET}")
        
        # Get broker status
        broker_status = self.broker.get_broker_status()
        if broker_status.get('order_summary'):
            summary = broker_status['order_summary']
            print(f"\nOrder Summary:")
            print(f"  Active Orders: {summary.get('active_orders', 0)}")
            print(f"  Today's Orders: {summary.get('todays_orders', 0)}")
    
    async def run(self):
        """Run all status checks"""
        print(f"{BLUE}QuantumSentiment Trading Bot - System Status{RESET}")
        print(f"{BLUE}{'='*50}{RESET}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Initialize
        if not await self.initialize():
            return
        
        # Display all sections
        await self.display_account_info()
        await self.display_positions()
        await self.display_recent_orders()
        await self.display_performance_metrics()
        await self.display_system_health()
        
        print(f"\n{BLUE}{'='*50}{RESET}")
        print("Status check complete")


async def main():
    """Main entry point"""
    checker = StatusChecker()
    await checker.run()


if __name__ == "__main__":
    asyncio.run(main())