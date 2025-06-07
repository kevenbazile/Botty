import time
import sys
from strategy import simple_strategy

# Jesse integration (optional)
try:
    from jesse_integration import run_backtest, run_optimization, create_jesse_strategy_template
    from datetime import datetime, timedelta
    JESSE_AVAILABLE = True
except ImportError:
    JESSE_AVAILABLE = False

def run():
    print("🚀 Starting bot...")
    while True:
        simple_strategy()
        time.sleep(10)  # Run every 10 seconds

def run_jesse_backtest(days=30):
    """Run Jesse-style backtest"""
    if not JESSE_AVAILABLE:
        print("❌ Jesse components not available")
        return
    
    print(f"🧪 Running {days}-day backtest...")
    
    # Create a simple args object
    class Args:
        def __init__(self):
            self.exchange = 'kraken'
            self.symbol = 'BTC/USD'
            self.timeframe = '1h'
            self.strategy = 'golden_cross'
            self.days = days
            self.start_date = None
            self.end_date = None
            self.balance = 10000.0
            self.fee_rate = 0.001
    
    args = Args()
    return run_backtest(args)

def run_jesse_optimize(days=60, generations=10):
    """Run Jesse-style optimization"""
    if not JESSE_AVAILABLE:
        print("❌ Jesse components not available")
        return
    
    print(f"🧬 Running optimization on {days} days with {generations} generations...")
    
    class Args:
        def __init__(self):
            self.exchange = 'kraken'
            self.symbol = 'BTC/USD'
            self.timeframe = '1h'
            self.strategy = 'golden_cross'
            self.days = days
            self.balance = 10000.0
            self.fee_rate = 0.001
            self.generations = generations
            self.population = 20
    
    args = Args()
    return run_optimization(args)

def main():
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "backtest":
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            run_jesse_backtest(days)
            
        elif command == "optimize":
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 60
            generations = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            run_jesse_optimize(days, generations)
            
        elif command == "template":
            if JESSE_AVAILABLE:
                create_jesse_strategy_template()
            else:
                print("❌ Jesse components not available")
                
        elif command == "help":
            print("📋 Available commands:")
            print("  python main.py                    - Run live trading bot")
            print("  python main.py backtest [days]    - Run backtest (default: 30 days)")
            print("  python main.py optimize [days] [gen] - Optimize strategy (default: 60 days, 10 gen)")
            print("  python main.py template           - Create strategy template")
            print("  python main.py help               - Show this help")
            
        else:
            print(f"❌ Unknown command: {command}")
            print("Use 'python main.py help' for available commands")
    else:
        # Default: run live trading
        run()

if __name__ == "__main__":
    main()