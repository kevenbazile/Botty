import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt
import json
from dataclasses import dataclass
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class Trade:
    """Represents a single trade"""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    side: str  # 'buy' or 'sell'
    pnl: Optional[float] = None
    fees: float = 0.0
    
    @property
    def is_open(self) -> bool:
        return self.exit_time is None
    
    def close_trade(self, exit_price: float, exit_time: datetime, fee: float = 0.0):
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.fees += fee
        
        if self.side == 'buy':
            self.pnl = (exit_price - self.entry_price) * self.quantity - self.fees
        else:  # sell/short
            self.pnl = (self.entry_price - exit_price) * self.quantity - self.fees

@dataclass
class Portfolio:
    """Portfolio state management"""
    initial_balance: float = 10000.0
    current_balance: float = 10000.0
    positions: Dict = None
    trades: List[Trade] = None
    
    def __post_init__(self):
        if self.positions is None:
            self.positions = {}
        if self.trades is None:
            self.trades = []
    
    def calculate_equity(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio equity"""
        equity = self.current_balance
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                equity += position['quantity'] * current_prices[symbol]
        return equity

class DataFetcher:
    """Fetch historical data for backtesting"""
    
    def __init__(self, exchange_name: str = 'kraken'):
        self.exchange = getattr(ccxt, exchange_name)()
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', 
                   since: datetime = None, limit: int = 1000) -> pd.DataFrame:
        """Fetch OHLCV data"""
        try:
            if since:
                since_ms = int(since.timestamp() * 1000)
            else:
                since_ms = None
                
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since_ms, limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

class TechnicalIndicators:
    """Common technical indicators for strategies"""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2):
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD Indicator"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

class BacktestStrategy:
    """Base strategy class - override the should_buy/should_sell methods"""
    
    def __init__(self, name: str = "Base Strategy"):
        self.name = name
        self.indicators = TechnicalIndicators()
    
    def should_buy(self, data: pd.DataFrame, current_idx: int) -> bool:
        """Override this method with your buy logic"""
        return False
    
    def should_sell(self, data: pd.DataFrame, current_idx: int, position) -> bool:
        """Override this method with your sell logic"""
        return False
    
    def calculate_position_size(self, balance: float, price: float) -> float:
        """Calculate position size - default to 10% of balance"""
        return (balance * 0.1) / price

class SimpleMovingAverageStrategy(BacktestStrategy):
    """Example strategy: SMA crossover"""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 20):
        super().__init__("SMA Crossover Strategy")
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def should_buy(self, data: pd.DataFrame, current_idx: int) -> bool:
        if current_idx < self.slow_period:
            return False
            
        fast_sma = self.indicators.sma(data['close'], self.fast_period).iloc[current_idx]
        slow_sma = self.indicators.sma(data['close'], self.slow_period).iloc[current_idx]
        
        # Buy when fast SMA crosses above slow SMA
        prev_fast = self.indicators.sma(data['close'], self.fast_period).iloc[current_idx-1]
        prev_slow = self.indicators.sma(data['close'], self.slow_period).iloc[current_idx-1]
        
        return (fast_sma > slow_sma) and (prev_fast <= prev_slow)
    
    def should_sell(self, data: pd.DataFrame, current_idx: int, position) -> bool:
        if current_idx < self.slow_period:
            return False
            
        fast_sma = self.indicators.sma(data['close'], self.fast_period).iloc[current_idx]
        slow_sma = self.indicators.sma(data['close'], self.slow_period).iloc[current_idx]
        
        # Sell when fast SMA crosses below slow SMA
        prev_fast = self.indicators.sma(data['close'], self.fast_period).iloc[current_idx-1]
        prev_slow = self.indicators.sma(data['close'], self.slow_period).iloc[current_idx-1]
        
        return (fast_sma < slow_sma) and (prev_fast >= prev_slow)

class Backtester:
    """Main backtesting engine"""
    
    def __init__(self, initial_balance: float = 10000.0, fee_rate: float = 0.001):
        self.portfolio = Portfolio(initial_balance=initial_balance)
        self.fee_rate = fee_rate
        self.results = {}
    
    def run_backtest(self, data: pd.DataFrame, strategy: BacktestStrategy, symbol: str = 'BTC/USD'):
        """Run the backtest"""
        print(f"🚀 Running backtest for {strategy.name}")
        print(f"📊 Data period: {data.index[0]} to {data.index[-1]}")
        print(f"💰 Initial balance: ${self.portfolio.initial_balance:,.2f}")
        
        for i in range(len(data)):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            
            # Check for buy signals
            if symbol not in self.portfolio.positions and strategy.should_buy(data, i):
                quantity = strategy.calculate_position_size(self.portfolio.current_balance, current_price)
                cost = quantity * current_price
                fee = cost * self.fee_rate
                
                if cost + fee <= self.portfolio.current_balance:
                    # Execute buy
                    trade = Trade(
                        entry_time=current_time,
                        entry_price=current_price,
                        quantity=quantity,
                        side='buy',
                        fees=fee
                    )
                    
                    self.portfolio.trades.append(trade)
                    self.portfolio.positions[symbol] = {
                        'quantity': quantity,
                        'entry_price': current_price,
                        'trade_id': len(self.portfolio.trades) - 1
                    }
                    self.portfolio.current_balance -= (cost + fee)
                    
                    print(f"🟢 BUY: {quantity:.6f} {symbol} at ${current_price:.2f} ({current_time})")
            
            # Check for sell signals
            elif symbol in self.portfolio.positions and strategy.should_sell(data, i, self.portfolio.positions[symbol]):
                position = self.portfolio.positions[symbol]
                trade_id = position['trade_id']
                quantity = position['quantity']
                
                revenue = quantity * current_price
                fee = revenue * self.fee_rate
                
                # Close the trade
                self.portfolio.trades[trade_id].close_trade(current_price, current_time, fee)
                self.portfolio.current_balance += (revenue - fee)
                
                pnl = self.portfolio.trades[trade_id].pnl
                print(f"🔴 SELL: {quantity:.6f} {symbol} at ${current_price:.2f} | PnL: ${pnl:.2f} ({current_time})")
                
                del self.portfolio.positions[symbol]
        
        self._calculate_results(data, symbol)
        return self.results
    
    def _calculate_results(self, data: pd.DataFrame, symbol: str):
        """Calculate backtest results and metrics"""
        closed_trades = [t for t in self.portfolio.trades if not t.is_open]
        
        if not closed_trades:
            print("❌ No completed trades found")
            return
        
        # Basic metrics
        total_trades = len(closed_trades)
        winning_trades = len([t for t in closed_trades if t.pnl > 0])
        losing_trades = total_trades - winning_trades
        
        total_pnl = sum(t.pnl for t in closed_trades)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Calculate final equity
        current_prices = {symbol: data['close'].iloc[-1]}
        final_equity = self.portfolio.calculate_equity(current_prices)
        total_return = ((final_equity - self.portfolio.initial_balance) / self.portfolio.initial_balance) * 100
        
        # Advanced metrics
        returns = [t.pnl / self.portfolio.initial_balance for t in closed_trades]
        sharpe_ratio = self._calculate_sharpe_ratio(returns) if len(returns) > 1 else 0
        max_drawdown = self._calculate_max_drawdown(closed_trades)
        
        self.results = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'final_equity': final_equity,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades': closed_trades
        }
        
        self._print_results()
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        return (mean_return / std_return) * np.sqrt(252) if std_return != 0 else 0
    
    def _calculate_max_drawdown(self, trades: List[Trade]) -> float:
        """Calculate maximum drawdown"""
        cumulative_pnl = 0
        peak = 0
        max_dd = 0
        
        for trade in trades:
            cumulative_pnl += trade.pnl
            if cumulative_pnl > peak:
                peak = cumulative_pnl
            drawdown = peak - cumulative_pnl
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    def _print_results(self):
        """Print formatted results"""
        print("\n" + "="*50)
        print("📈 BACKTEST RESULTS")
        print("="*50)
        print(f"Total Trades: {self.results['total_trades']}")
        print(f"Winning Trades: {self.results['winning_trades']}")
        print(f"Losing Trades: {self.results['losing_trades']}")
        print(f"Win Rate: {self.results['win_rate']:.2f}%")
        print(f"Total PnL: ${self.results['total_pnl']:.2f}")
        print(f"Total Return: {self.results['total_return']:.2f}%")
        print(f"Final Equity: ${self.results['final_equity']:.2f}")
        print(f"Sharpe Ratio: {self.results['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: ${self.results['max_drawdown']:.2f}")
        print("="*50)
    
    def plot_results(self, data: pd.DataFrame):
        """Plot backtest results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Price chart with trades
        ax1 = axes[0, 0]
        ax1.plot(data.index, data['close'], label='Price', alpha=0.7)
        
        for trade in self.portfolio.trades:
            if not trade.is_open:
                ax1.scatter(trade.entry_time, trade.entry_price, color='green', marker='^', s=100)
                ax1.scatter(trade.exit_time, trade.exit_price, color='red', marker='v', s=100)
        
        ax1.set_title('Price Chart with Trade Signals')
        ax1.set_ylabel('Price')
        ax1.legend()
        
        # PnL chart
        ax2 = axes[0, 1]
        pnl_values = [t.pnl for t in self.portfolio.trades if not t.is_open]
        ax2.bar(range(len(pnl_values)), pnl_values, color=['green' if p > 0 else 'red' for p in pnl_values])
        ax2.set_title('Trade PnL')
        ax2.set_ylabel('PnL ($)')
        ax2.set_xlabel('Trade Number')
        
        # Cumulative returns
        ax3 = axes[1, 0]
        cumulative_pnl = np.cumsum(pnl_values)
        ax3.plot(cumulative_pnl, color='blue')
        ax3.set_title('Cumulative PnL')
        ax3.set_ylabel('Cumulative PnL ($)')
        ax3.set_xlabel('Trade Number')
        
        # Win/Loss distribution
        ax4 = axes[1, 1]
        wins = len([p for p in pnl_values if p > 0])
        losses = len([p for p in pnl_values if p <= 0])
        ax4.pie([wins, losses], labels=['Wins', 'Losses'], colors=['green', 'red'], autopct='%1.1f%%')
        ax4.set_title('Win/Loss Distribution')
        
        plt.tight_layout()
        plt.show()

# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the backtesting framework
    
    # 1. Fetch historical data
    fetcher = DataFetcher('kraken')
    start_date = datetime.now() - timedelta(days=30)
    data = fetcher.fetch_ohlcv('BTC/USD', '1h', start_date, 500)
    
    if not data.empty:
        # 2. Create a strategy
        strategy = SimpleMovingAverageStrategy(fast_period=10, slow_period=20)
        
        # 3. Run backtest
        backtester = Backtester(initial_balance=10000, fee_rate=0.001)
        results = backtester.run_backtest(data, strategy, 'BTC/USD')
        
        # 4. Plot results (optional)
        # backtester.plot_results(data)
    else:
        print("❌ Could not fetch historical data")