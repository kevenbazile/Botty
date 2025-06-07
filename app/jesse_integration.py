#!/usr/bin/env python3
"""
Jesse-style trading bot integration
Combines your existing live trading with Jesse's backtesting capabilities
"""

import argparse
import sys
from datetime import datetime, timedelta
from jesse_strategy import Strategy, ta, utils
from example_jesse_strategy import GoldenCrossStrategy, AdvancedTrendStrategy
from jesse_backtesting_engine import JesseBacktester, JesseOptimizer

def run_live_trading():
    """Run your existing live trading bot with Jesse-style strategy"""
    print("🚀 Starting Jesse-style live trading...")
    
    # Use your existing strategy but with Jesse structure
    strategy = GoldenCrossStrategy('kraken', 'BTC/USD', '1h')
    
    # Your existing live trading loop
    import time
    while True:
        try:
            # Fetch current market data (use your existing method)
            # strategy.update_with_live_data()
            
            # Run Jesse-style decision making
            # This integrates with your existing simple_strategy()
            print("📊 Analyzing market with Jesse-style indicators...")
            
            # Example integration
            print(f"💰 Current Price: ${strategy.price:.2f}")
            print(f"📈 EMA Fast: ${strategy.ema_fast:.2f}")
            print(f"📈 EMA Slow: ${strategy.ema_slow:.2f}")
            print(f"📊 RSI: {strategy.rsi:.2f}")
            print(f"📊 Trend: {'Bullish' if strategy.trend == 1 else 'Bearish' if strategy.trend == -1 else 'Sideways'}")
            
            # Jesse-style entry logic
            if strategy.should_long():
                print("🟢 JESSE SIGNAL: LONG")
            elif strategy.should_short():
                print("🔴 JESSE SIGNAL: SHORT")
            else:
                print("⏸️ JESSE SIGNAL: HOLD")
            
            print("-" * 50)
            time.sleep(10)  # Your existing 10-second interval
            
        except KeyboardInterrupt:
            print("\n⏹️ Live trading stopped")
            break
        except Exception as e:
            print(f"❌ Error in live trading: {e}")
            time.sleep(10)

def run_backtest(args):
    """Run Jesse-style backtest"""
    print("🧪 Starting Jesse-style backtest...")
    
    # Parse dates
    if args.days:
        start_date = datetime.now() - timedelta(days=args.days)
        end_date = datetime.now()
    else:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else datetime.now() - timedelta(days=30)
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else datetime.now()
    
    # Choose strategy
    strategy_map = {
        'golden_cross': GoldenCrossStrategy,
        'advanced': AdvancedTrendStrategy
    }
    
    strategy_class = strategy_map.get(args.strategy, GoldenCrossStrategy)
    
    # Create backtester
    backtester = JesseBacktester(
        starting_balance=args.balance,
        fee_rate=args.fee_rate
    )
    
    # Run backtest
    results = backtester.run_backtest(
        strategy_class,
        args.exchange,
        args.symbol,
        args.timeframe,
        start_date,
        end_date
    )
    
    return results

def run_optimization(args):
    """Run Jesse-style optimization"""
    print("🧬 Starting Jesse-style optimization...")
    
    # Parse dates
    start_date = datetime.now() - timedelta(days=args.days)
    end_date = datetime.now()
    
    # Choose strategy
    strategy_map = {
        'golden_cross': GoldenCrossStrategy,
        'advanced': AdvancedTrendStrategy
    }
    
    strategy_class = strategy_map.get(args.strategy, GoldenCrossStrategy)
    
    # Create backtester and optimizer
    backtester = JesseBacktester(
        starting_balance=args.balance,
        fee_rate=args.fee_rate
    )
    
    optimizer = JesseOptimizer(backtester)
    
    # Run optimization
    results = optimizer.optimize(
        strategy_class,
        args.exchange,
        args.symbol,
        args.timeframe,
        start_date,
        end_date,
        generations=args.generations,
        population_size=args.population
    )
    
    return results

def create_jesse_strategy_template():
    """Create a template for new Jesse-style strategies"""
    template = '''from jesse_strategy import Strategy, ta, utils

class MyCustomStrategy(Strategy):
    """
    Custom Jesse-style strategy
    Follow Jesse conventions for best results
    """
    
    def __init__(self, exchange_name: str = 'kraken', symbol: str = 'BTC/USD', timeframe: str = '1h'):
        super().__init__(exchange_name, symbol, timeframe)
        # Strategy parameters
        self.fast_period = 20
        self.slow_period = 50
        self.risk_per_trade = 2.0
    
    # Jesse-style indicator properties
    @property
    def fast_ma(self):
        return ta.ema(self.candles, self.fast_period)
    
    @property
    def slow_ma(self):
        return ta.ema(self.candles, self.slow_period)
    
    @property
    def rsi_value(self):
        return ta.rsi(self.candles, 14)
    
    @property
    def atr_value(self):
        return ta.atr(self.candles, 14)
    
    @property
    def trend(self):
        if self.fast_ma > self.slow_ma:
            return 1  # Bullish
        elif self.fast_ma < self.slow_ma:
            return -1  # Bearish
        return 0  # Sideways
    
    def should_long(self) -> bool:
        """Long entry conditions"""
        return (
            self.trend == 1 and
            self.rsi_value < 70 and
            self.price > self.fast_ma
        )
    
    def should_short(self) -> bool:
        """Short entry conditions"""
        return (
            self.trend == -1 and
            self.rsi_value > 30 and
            self.price < self.fast_ma
        )
    
    def go_long(self):
        """Execute long position"""
        entry_price = self.price
        stop_loss_price = entry_price - (self.atr_value * 2)
        
        qty = utils.risk_to_qty(
            self.available_margin, 
            self.risk_per_trade, 
            entry_price, 
            stop_loss_price, 
            fee_rate=self.fee_rate
        )
        
        self.buy = qty, entry_price
    
    def go_short(self):
        """Execute short position"""
        entry_price = self.price
        stop_loss_price = entry_price + (self.atr_value * 2)
        
        qty = utils.risk_to_qty(
            self.available_margin, 
            self.risk_per_trade, 
            entry_price, 
            stop_loss_price, 
            fee_rate=self.fee_rate
        )
        
        self.sell = qty, entry_price
    
    def should_cancel_entry(self) -> bool:
        return True
    
    def on_open_position(self, order) -> None:
        """Set stop loss and take profit"""
        if self.is_long:
            self.stop_loss = self.position.qty, self.position.entry_price - (self.atr_value * 2)
            self.take_profit = self.position.qty, self.position.entry_price + (self.atr_value * 3)
        elif self.is_short:
            self.stop_loss = self.position.qty, self.position.entry_price + (self.atr_value * 2)
            self.take_profit = self.position.qty, self.position.entry_price - (self.atr_value * 3)
    
    def update_position(self) -> None:
        """Update position logic"""
        if self.is_long and self.trend == -1:
            self.liquidate()
        elif self.is_short and self.trend == 1:
            self.liquidate()
    
    def hyperparameters(self) -> list:
        """Parameters for optimization"""
        return [
            {'name': 'fast_period', 'type': int, 'min': 10, 'max': 30, 'default': 20},
            {'name': 'slow_period', 'type': int, 'min': 30, 'max': 100, 'default': 50},
            {'name': 'risk_per_trade', 'type': float, 'min': 1.0, 'max': 5.0, 'default': 2.0},
        ]
'''
    
    with open('my_jesse_strategy.py', 'w') as f:
        f.write(template)
    
    print("✅ Jesse strategy template created: my_jesse_strategy.py")
    print("📝 Edit this file to create your custom strategy")

def main():
    """Main CLI interface for Jesse integration"""
    parser = argparse.ArgumentParser(description='Jesse-style Trading Bot')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Live trading command
    live_parser = subparsers.add_parser('live', help='Run live trading with Jesse strategy')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run Jesse-style backtest')
    backtest_parser.add_argument('--exchange', type=str, default='kraken', help='Exchange name')
    backtest_parser.add_argument('--symbol', type=str, default='BTC/USD', help='Trading symbol')
    backtest_parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe')
    backtest_parser.add_argument('--strategy', type=str, default='golden_cross', 
                                choices=['golden_cross', 'advanced'], help='Strategy to test')
    backtest_parser.add_argument('--days', type=int, help='Number of days to backtest')
    backtest_parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--balance', type=float, default=10000, help='Starting balance')
    backtest_parser.add_argument('--fee-rate', type=float, default=0.001, help='Fee rate')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Run Jesse-style optimization')
    optimize_parser.add_argument('--exchange', type=str, default='kraken', help='Exchange name')
    optimize_parser.add_argument('--symbol', type=str, default='BTC/USD', help='Trading symbol')
    optimize_parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe')
    optimize_parser.add_argument('--strategy', type=str, default='golden_cross', 
                                choices=['golden_cross', 'advanced'], help='Strategy to optimize')
    optimize_parser.add_argument('--days', type=int, default=30, help='Number of days for optimization')
    optimize_parser.add_argument('--balance', type=float, default=10000, help='Starting balance')
    optimize_parser.add_argument('--fee-rate', type=float, default=0.001, help='Fee rate')
    optimize_parser.add_argument('--generations', type=int, default=10, help='Number of generations')
    optimize_parser.add_argument('--population', type=int, default=20, help='Population size')
    
    # Template command
    template_parser = subparsers.add_parser('template', help='Create Jesse strategy template')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'live':
            run_live_trading()
        elif args.command == 'backtest':
            run_backtest(args)
        elif args.command == 'optimize':
            run_optimization(args)
        elif args.command == 'template':
            create_jesse_strategy_template()
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n⏹️ Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        if '--debug' in sys.argv:
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()