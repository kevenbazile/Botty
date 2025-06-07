import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import ccxt
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class BacktestResults:
    """Jesse-style backtest results"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    starting_balance: float = 10000.0
    finishing_balance: float = 10000.0
    total_trading_volume: float = 0.0
    total_fees: float = 0.0
    largest_winning_trade: float = 0.0
    largest_losing_trade: float = 0.0
    average_winning_trade: float = 0.0
    average_losing_trade: float = 0.0
    ratio_avg_win_loss: float = 0.0

class JesseBacktester:
    """
    Jesse-compatible backtesting engine
    """
    
    def __init__(self, starting_balance: float = 10000.0, fee_rate: float = 0.001):
        self.starting_balance = starting_balance
        self.fee_rate = fee_rate
        self.results = BacktestResults()
        
    def fetch_candles(self, exchange_name: str, symbol: str, timeframe: str, 
                     start_date: datetime, end_date: datetime = None) -> np.ndarray:
        """Fetch historical candles for backtesting"""
        try:
            exchange = getattr(ccxt, exchange_name)()
            
            if end_date is None:
                end_date = datetime.now()
            
            # Calculate required number of candles
            timeframe_minutes = self._timeframe_to_minutes(timeframe)
            total_minutes = int((end_date - start_date).total_seconds() / 60)
            limit = min(total_minutes // timeframe_minutes, 1000)
            
            since = int(start_date.timestamp() * 1000)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            
            # Convert to numpy array: [timestamp, open, close, high, low, volume]
            candles = np.array(ohlcv)
            
            # Jesse format: [timestamp, open, close, high, low, volume]
            if len(candles) > 0:
                # Rearrange columns to match Jesse format
                jesse_candles = np.column_stack([
                    candles[:, 0],  # timestamp
                    candles[:, 1],  # open
                    candles[:, 4],  # close
                    candles[:, 2],  # high
                    candles[:, 3],  # low
                    candles[:, 5]   # volume
                ])
                return jesse_candles
            
            return np.array([])
            
        except Exception as e:
            print(f"Error fetching candles: {e}")
            return np.array([])
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        timeframe_map = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080
        }
        return timeframe_map.get(timeframe, 60)
    
    def run_backtest(self, strategy_class, exchange_name: str, symbol: str, 
                    timeframe: str, start_date: datetime, end_date: datetime = None) -> BacktestResults:
        """
        Run Jesse-style backtest
        """
        print(f"🚀 Running Jesse-style backtest...")
        print(f"📊 Strategy: {strategy_class.__name__}")
        print(f"💱 Exchange: {exchange_name}")
        print(f"📈 Symbol: {symbol}")
        print(f"⏰ Timeframe: {timeframe}")
        print(f"📅 Period: {start_date.date()} to {(end_date or datetime.now()).date()}")
        
        # Fetch historical data
        candles = self.fetch_candles(exchange_name, symbol, timeframe, start_date, end_date)
        
        if len(candles) == 0:
            print("❌ No candle data available")
            return self.results
        
        print(f"✅ Loaded {len(candles)} candles")
        
        # Initialize strategy
        strategy = strategy_class(exchange_name, symbol, timeframe)
        strategy.balance = self.starting_balance
        strategy.available_margin = self.starting_balance
        strategy.is_backtesting = True
        strategy.fee_rate = self.fee_rate
        
        # Track performance
        balance_history = [self.starting_balance]
        trade_history = []
        
        # Run backtest
        for i in range(len(candles)):
            # Update strategy state
            strategy.candles = candles[:i+1] if i > 0 else candles[:1]
            strategy.current_candle = candles[i]
            strategy.index = i
            
            # Reset orders for this candle
            strategy.buy = None
            strategy.sell = None
            strategy.stop_loss = None
            strategy.take_profit = None
            
            # Jesse workflow: before() -> position logic -> after()
            strategy.before()
            
            # Update existing position
            if strategy.position.is_open:
                strategy.position.update(strategy.price)
                strategy.update_position()
                
                # Check stop loss and take profit
                self._check_stop_loss_take_profit(strategy)
                
            else:
                # Check for new entries
                if strategy.should_cancel_entry():
                    # Cancel any pending orders
                    pass
                
                # Check entry conditions and filters
                filters_pass = all(filter_func() for filter_func in strategy.filters())
                
                if filters_pass:
                    if strategy.should_long():
                        strategy.go_long()
                        if strategy.buy:
                            self._execute_order(strategy, 'buy', trade_history)
                    
                    elif strategy.should_short():
                        strategy.go_short()
                        if strategy.sell:
                            self._execute_order(strategy, 'sell', trade_history)
            
            # Track balance
            current_balance = self._calculate_current_balance(strategy)
            balance_history.append(current_balance)
            
            strategy.after()
        
        # Calculate results
        self._calculate_results(trade_history, balance_history)
        self._print_results()
        
        return self.results
    
    def _execute_order(self, strategy, order_type: str, trade_history: List):
        """Execute buy/sell orders"""
        try:
            if order_type == 'buy' and strategy.buy:
                qty, price = strategy.buy
                
                # Calculate cost with fees
                cost = qty * price
                fee = cost * strategy.fee_rate
                total_cost = cost + fee
                
                if total_cost <= strategy.available_margin:
                    # Open long position
                    strategy.position.entry_price = price
                    strategy.position.qty = qty
                    strategy.position.type = 'long'
                    strategy.position.is_open = True
                    strategy.position.opened_at = strategy.current_candle[0]
                    
                    # Update balance
                    strategy.available_margin -= total_cost
                    
                    # Record trade start
                    trade_record = {
                        'type': 'long',
                        'entry_price': price,
                        'entry_time': strategy.current_candle[0],
                        'qty': qty,
                        'fee': fee
                    }
                    trade_history.append(trade_record)
                    
                    # Call position opened event
                    strategy.on_open_position(None)
                    
                    print(f"📈 LONG opened: {qty:.4f} @ ${price:.2f} (Fee: ${fee:.2f})")
            
            elif order_type == 'sell' and strategy.sell:
                qty, price = strategy.sell
                
                # Calculate cost with fees
                cost = qty * price
                fee = cost * strategy.fee_rate
                total_cost = cost + fee
                
                if total_cost <= strategy.available_margin:
                    # Open short position
                    strategy.position.entry_price = price
                    strategy.position.qty = qty
                    strategy.position.type = 'short'
                    strategy.position.is_open = True
                    strategy.position.opened_at = strategy.current_candle[0]
                    
                    # Update balance
                    strategy.available_margin -= total_cost
                    
                    # Record trade start
                    trade_record = {
                        'type': 'short',
                        'entry_price': price,
                        'entry_time': strategy.current_candle[0],
                        'qty': qty,
                        'fee': fee
                    }
                    trade_history.append(trade_record)
                    
                    # Call position opened event
                    strategy.on_open_position(None)
                    
                    print(f"📉 SHORT opened: {qty:.4f} @ ${price:.2f} (Fee: ${fee:.2f})")
                    
        except Exception as e:
            print(f"❌ Error executing order: {e}")
    
    def _check_stop_loss_take_profit(self, strategy):
        """Check and execute stop loss / take profit"""
        if not strategy.position.is_open:
            return
        
        current_price = strategy.price
        
        # Check stop loss and take profit based on strategy orders
        if hasattr(strategy, 'stop_loss') and strategy.stop_loss:
            stop_qty, stop_price = strategy.stop_loss
            
            if strategy.position.type == 'long':
                # For long positions, stop loss triggers if price goes below stop
                if current_price <= stop_price:
                    self._close_position(strategy, current_price, 'Stop Loss')
                    return
            
            elif strategy.position.type == 'short':
                # For short positions, stop loss triggers if price goes above stop
                if current_price >= stop_price:
                    self._close_position(strategy, current_price, 'Stop Loss')
                    return
        
        # Check take profit
        if hasattr(strategy, 'take_profit') and strategy.take_profit:
            tp_qty, tp_price = strategy.take_profit
            
            if strategy.position.type == 'long':
                # For long positions, take profit triggers if price goes above target
                if current_price >= tp_price:
                    self._close_position(strategy, current_price, 'Take Profit')
                    return
            
            elif strategy.position.type == 'short':
                # For short positions, take profit triggers if price goes below target
                if current_price <= tp_price:
                    self._close_position(strategy, current_price, 'Take Profit')
                    return
    
    def _close_position(self, strategy, exit_price: float, reason: str = 'Manual'):
        """Close the current position"""
        if not strategy.position.is_open:
            return
        
        qty = strategy.position.qty
        entry_price = strategy.position.entry_price
        
        # Calculate PnL
        if strategy.position.type == 'long':
            pnl = (exit_price - entry_price) * qty
        else:  # short
            pnl = (entry_price - exit_price) * qty
        
        # Calculate fees
        exit_cost = qty * exit_price
        fee = exit_cost * strategy.fee_rate
        net_pnl = pnl - fee
        
        # Update balance
        strategy.available_margin += exit_cost - fee
        strategy.balance += net_pnl
        
        # Update position
        strategy.position.pnl = net_pnl
        strategy.position.update(exit_price)
        
        # Update trade history with complete trade data
        # Find the last incomplete trade record
        for trade in reversed(getattr(self, '_current_trades', [])):
            if (trade.get('type') == strategy.position.type and 
                trade.get('entry_price') == entry_price and 
                'exit_price' not in trade):
                
                trade.update({
                    'exit_price': exit_price,
                    'exit_time': strategy.current_candle[0],
                    'pnl': net_pnl,
                    'exit_fee': fee,
                    'reason': reason
                })
                break
        
        # Close position
        strategy.position.is_open = False
        strategy.position.type = 'close'
        
        # Call position closed event
        strategy.on_close_position(None)
        
        print(f"🔒 Position closed @ ${exit_price:.2f} | PnL: ${net_pnl:.2f} | Reason: {reason}")
    
    def run_backtest(self, strategy_class, exchange_name: str, symbol: str, 
                    timeframe: str, start_date: datetime, end_date: datetime = None) -> BacktestResults:
        """
        Run Jesse-style backtest
        """
        print(f"🚀 Running Jesse-style backtest...")
        print(f"📊 Strategy: {strategy_class.__name__}")
        print(f"💱 Exchange: {exchange_name}")
        print(f"📈 Symbol: {symbol}")
        print(f"⏰ Timeframe: {timeframe}")
        print(f"📅 Period: {start_date.date()} to {(end_date or datetime.now()).date()}")
        
        # Fetch historical data
        candles = self.fetch_candles(exchange_name, symbol, timeframe, start_date, end_date)
        
        if len(candles) == 0:
            print("❌ No candle data available")
            return self.results
        
        print(f"✅ Loaded {len(candles)} candles")
        
        # Initialize strategy
        strategy = strategy_class(exchange_name, symbol, timeframe)
        strategy.balance = self.starting_balance
        strategy.available_margin = self.starting_balance
        strategy.is_backtesting = True
        strategy.fee_rate = self.fee_rate
        
        # Track performance
        balance_history = [self.starting_balance]
        trade_history = []
        self._current_trades = trade_history  # Make trade_history accessible for _close_position
        
        # Run backtest
        for i in range(len(candles)):
            # Update strategy state
            strategy.candles = candles[:i+1] if i > 0 else candles[:1]
            strategy.current_candle = candles[i]
            strategy.index = i
            
            # Reset orders for this candle
            strategy.buy = None
            strategy.sell = None
            strategy.stop_loss = None
            strategy.take_profit = None
            
            # Jesse workflow: before() -> position logic -> after()
            strategy.before()
            
            # Update existing position
            if strategy.position.is_open:
                strategy.position.update(strategy.price)
                strategy.update_position()
                
                # Check stop loss and take profit
                self._check_stop_loss_take_profit(strategy)
                
            else:
                # Check for new entries
                if strategy.should_cancel_entry():
                    # Cancel any pending orders
                    pass
                
                # Check entry conditions and filters
                filters_pass = all(filter_func() for filter_func in strategy.filters())
                
                if filters_pass:
                    if strategy.should_long():
                        strategy.go_long()
                        if strategy.buy:
                            self._execute_order(strategy, 'buy', trade_history)
                    
                    elif strategy.should_short():
                        strategy.go_short()
                        if strategy.sell:
                            self._execute_order(strategy, 'sell', trade_history)
            
            # Track balance
            current_balance = self._calculate_current_balance(strategy)
            balance_history.append(current_balance)
            
            strategy.after()
        
        # Close any remaining open positions
        if strategy.position.is_open:
            self._close_position(strategy, strategy.price, 'End of backtest')
        
        # Calculate results
        self._calculate_results(trade_history, balance_history)
        self._print_results()
        
        return self.results
    
    def _calculate_results(self, trade_history: List, balance_history: List):
        """Calculate comprehensive backtest results"""
        if len(balance_history) < 2:
            return
        
        # Filter only completed trades (have exit_price)
        completed_trades = [t for t in trade_history if 'exit_price' in t and 'pnl' in t]
        
        # Basic metrics
        self.results.starting_balance = balance_history[0]
        self.results.finishing_balance = balance_history[-1]
        self.results.total_return = ((self.results.finishing_balance - self.results.starting_balance) / self.results.starting_balance) * 100
        
        # Trade statistics
        profitable_trades = [t for t in completed_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in completed_trades if t.get('pnl', 0) <= 0]
        
        self.results.total_trades = len(completed_trades)
        self.results.winning_trades = len(profitable_trades)
        self.results.losing_trades = len(losing_trades)
        self.results.win_rate = (self.results.winning_trades / self.results.total_trades * 100) if self.results.total_trades > 0 else 0
        
        # Risk metrics
        self.results.max_drawdown = self._calculate_max_drawdown(balance_history)
        self.results.sharpe_ratio = self._calculate_sharpe_ratio(balance_history)
        
        # Trade analysis
        if profitable_trades:
            self.results.largest_winning_trade = max(t.get('pnl', 0) for t in profitable_trades)
            self.results.average_winning_trade = sum(t.get('pnl', 0) for t in profitable_trades) / len(profitable_trades)
        
        if losing_trades:
            self.results.largest_losing_trade = min(t.get('pnl', 0) for t in losing_trades)
            self.results.average_losing_trade = sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades)
        
        # Win/Loss ratio
        if self.results.average_losing_trade != 0:
            self.results.ratio_avg_win_loss = abs(self.results.average_winning_trade / self.results.average_losing_trade)
    
    def _calculate_max_drawdown(self, balance_history: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = balance_history[0]
        max_dd = 0
        
        for balance in balance_history:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak * 100
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, balance_history: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if len(balance_history) < 2:
            return 0
        
        returns = [
            (balance_history[i] - balance_history[i-1]) / balance_history[i-1]
            for i in range(1, len(balance_history))
        ]
        
        if len(returns) == 0:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        # Annualized Sharpe ratio (assuming daily returns)
        return (mean_return / std_return) * np.sqrt(365)
    
    def _print_results(self):
        """Print Jesse-style results"""
        print("\n" + "="*60)
        print("📊 JESSE-STYLE BACKTEST RESULTS")
        print("="*60)
        print(f"Starting Balance: ${self.results.starting_balance:,.2f}")
        print(f"Finishing Balance: ${self.results.finishing_balance:,.2f}")
        print(f"Total Return: {self.results.total_return:.2f}%")
        print(f"Max Drawdown: {self.results.max_drawdown:.2f}%")
        print(f"Sharpe Ratio: {self.results.sharpe_ratio:.3f}")
        print("")
        print("📈 TRADE STATISTICS:")
        print(f"Total Trades: {self.results.total_trades}")
        print(f"Winning Trades: {self.results.winning_trades}")
        print(f"Losing Trades: {self.results.losing_trades}")
        print(f"Win Rate: {self.results.win_rate:.2f}%")
        print("")
        if self.results.winning_trades > 0:
            print(f"Average Winning Trade: ${self.results.average_winning_trade:.2f}")
            print(f"Largest Winning Trade: ${self.results.largest_winning_trade:.2f}")
        if self.results.losing_trades > 0:
            print(f"Average Losing Trade: ${self.results.average_losing_trade:.2f}")
            print(f"Largest Losing Trade: ${self.results.largest_losing_trade:.2f}")
        if self.results.ratio_avg_win_loss > 0:
            print(f"Avg Win/Loss Ratio: {self.results.ratio_avg_win_loss:.2f}")
        print("="*60)

# Optimization Engine (simplified)
class JesseOptimizer:
    """
    Jesse-style hyperparameter optimization
    """
    
    def __init__(self, backtester: JesseBacktester):
        self.backtester = backtester
        self.best_results = None
        self.best_params = None
    
    def optimize(self, strategy_class, exchange_name: str, symbol: str, timeframe: str,
                start_date: datetime, end_date: datetime = None, 
                generations: int = 50, population_size: int = 20) -> Dict:
        """
        Run genetic algorithm optimization (simplified)
        """
        print(f"🧬 Starting Jesse-style optimization...")
        print(f"📊 Strategy: {strategy_class.__name__}")
        print(f"🔄 Generations: {generations}")
        print(f"👥 Population Size: {population_size}")
        
        # Get hyperparameters
        strategy_instance = strategy_class()
        hyperparams = strategy_instance.hyperparameters()
        
        if not hyperparams:
            print("❌ No hyperparameters defined for optimization")
            return {}
        
        best_return = -float('inf')
        
        for generation in range(generations):
            print(f"\n🔄 Generation {generation + 1}/{generations}")
            
            # Generate population
            population = self._generate_population(hyperparams, population_size)
            
            # Evaluate each individual
            for i, params in enumerate(population):
                # Create strategy with these parameters
                test_strategy = self._create_strategy_with_params(strategy_class, params)
                
                # Run backtest
                results = self.backtester.run_backtest(
                    lambda: test_strategy, exchange_name, symbol, timeframe, start_date, end_date
                )
                
                # Check if this is the best
                if results.total_return > best_return:
                    best_return = results.total_return
                    self.best_results = results
                    self.best_params = params.copy()
                    
                    print(f"✨ New best found! Return: {best_return:.2f}%")
                    print(f"📋 Parameters: {params}")
                
                if (i + 1) % 5 == 0:
                    print(f"⏳ Tested {i + 1}/{population_size} combinations...")
        
        print(f"\n🏆 Optimization complete!")
        print(f"📈 Best Return: {best_return:.2f}%")
        print(f"📋 Best Parameters: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_results': self.best_results,
            'best_return': best_return
        }
    
    def _generate_population(self, hyperparams: List[Dict], size: int) -> List[Dict]:
        """Generate random population for optimization"""
        population = []
        
        for _ in range(size):
            individual = {}
            for param in hyperparams:
                if isinstance(param['type'], type) and param['type'] is int:
                    individual[param['name']] = np.random.randint(param['min'], param['max'] + 1)
                elif isinstance(param['type'], type) and param['type'] is float:
                    individual[param['name']] = np.random.uniform(param['min'], param['max'])
                elif param['type'] == 'categorical':
                    individual[param['name']] = np.random.choice(param['options'])
            
            population.append(individual)
        
        return population
    
    def _create_strategy_with_params(self, strategy_class, params: Dict):
        """Create strategy instance with specific parameters"""
        strategy = strategy_class()
        
        # Set hyperparameters
        for key, value in params.items():
            if hasattr(strategy, key):
                setattr(strategy, key, value)
        
        return strategy