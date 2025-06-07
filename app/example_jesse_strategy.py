from jesse_strategy import Strategy, ta, utils

class GoldenCrossStrategy(Strategy):
    """
    Jesse-style Golden Cross strategy with EMA crossover
    """
    
    def __init__(self, exchange_name: str = 'kraken', symbol: str = 'BTC/USD', timeframe: str = '1h'):
        super().__init__(exchange_name, symbol, timeframe)
        # Strategy parameters (can be optimized)
        self.fast_period = 20
        self.slow_period = 50
        self.rsi_period = 14
        self.atr_period = 14
        self.risk_per_trade = 2.0  # 2% risk per trade
    
    # Jesse-style indicator properties
    @property
    def ema_fast(self):
        """Fast EMA"""
        return ta.ema(self.candles, self.fast_period)
    
    @property
    def ema_slow(self):
        """Slow EMA"""
        return ta.ema(self.candles, self.slow_period)
    
    @property
    def rsi(self):
        """RSI indicator"""
        return ta.rsi(self.candles, self.rsi_period)
    
    @property
    def atr(self):
        """Average True Range"""
        return ta.atr(self.candles, self.atr_period)
    
    @property
    def trend(self):
        """Determine trend direction"""
        if self.ema_fast > self.ema_slow:
            return 1  # Uptrend
        elif self.ema_fast < self.ema_slow:
            return -1  # Downtrend
        else:
            return 0  # Sideways
    
    # Entry conditions
    def should_long(self) -> bool:
        """Long entry conditions"""
        return (
            self.trend == 1 and  # Uptrend
            self.rsi < 70 and    # Not overbought
            self.price > self.ema_fast  # Price above fast EMA
        )
    
    def should_short(self) -> bool:
        """Short entry conditions"""
        return (
            self.trend == -1 and  # Downtrend
            self.rsi > 30 and     # Not oversold
            self.price < self.ema_fast  # Price below fast EMA
        )
    
    def go_long(self):
        """Execute long position"""
        entry_price = self.price
        stop_loss_price = entry_price - (self.atr * 2)  # 2 ATR stop loss
        
        # Calculate position size based on risk
        qty = utils.risk_to_qty(
            self.available_margin, 
            self.risk_per_trade, 
            entry_price, 
            stop_loss_price, 
            fee_rate=self.fee_rate
        )
        
        # Place market order
        self.buy = qty, entry_price
        self.log(f"Going LONG: {qty:.4f} @ ${entry_price:.2f}")
    
    def go_short(self):
        """Execute short position"""
        entry_price = self.price
        stop_loss_price = entry_price + (self.atr * 2)  # 2 ATR stop loss
        
        # Calculate position size based on risk
        qty = utils.risk_to_qty(
            self.available_margin, 
            self.risk_per_trade, 
            entry_price, 
            stop_loss_price, 
            fee_rate=self.fee_rate
        )
        
        # Place market order
        self.sell = qty, entry_price
        self.log(f"Going SHORT: {qty:.4f} @ ${entry_price:.2f}")
    
    def should_cancel_entry(self) -> bool:
        """Cancel pending orders if trend changes"""
        return True
    
    def on_open_position(self, order) -> None:
        """Set stop loss and take profit when position opens"""
        if self.is_long:
            stop_price = self.position.entry_price - (self.atr * 2)
            take_profit_price = self.position.entry_price + (self.atr * 3)
            
            self.stop_loss = self.position.qty, stop_price
            self.take_profit = self.position.qty, take_profit_price
            
            self.log(f"Position opened LONG @ ${self.position.entry_price:.2f}")
            self.log(f"Stop Loss: ${stop_price:.2f}, Take Profit: ${take_profit_price:.2f}")
            
        elif self.is_short:
            stop_price = self.position.entry_price + (self.atr * 2)  # Higher price for short stop
            take_profit_price = self.position.entry_price - (self.atr * 3)  # Lower price for short profit
            
            self.stop_loss = self.position.qty, stop_price
            self.take_profit = self.position.qty, take_profit_price
            
            self.log(f"Position opened SHORT @ ${self.position.entry_price:.2f}")
            self.log(f"Stop Loss: ${stop_price:.2f}, Take Profit: ${take_profit_price:.2f}")
    
    def update_position(self) -> None:
        """Update position (trailing stops, early exits)"""
        if self.is_long:
            # Exit if trend changes
            if self.trend == -1:
                self.liquidate()
                self.log("Exiting LONG: Trend changed")
                
        elif self.is_short:
            # Exit if trend changes
            if self.trend == 1:
                self.liquidate()
                self.log("Exiting SHORT: Trend changed")
    
    def filters(self) -> list:
        """Entry filters"""
        return [self.atr_filter, self.volume_filter]
    
    def atr_filter(self) -> bool:
        """Only trade when there's enough volatility"""
        return self.atr > self.price * 0.005  # ATR > 0.5% of price
    
    def volume_filter(self) -> bool:
        """Volume filter (simplified)"""
        # In real implementation, check volume
        return True
    
    def hyperparameters(self) -> list:
        """Parameters for optimization"""
        return [
            {'name': 'fast_period', 'type': int, 'min': 10, 'max': 30, 'default': 20},
            {'name': 'slow_period', 'type': int, 'min': 30, 'max': 100, 'default': 50},
            {'name': 'risk_per_trade', 'type': float, 'min': 1.0, 'max': 5.0, 'default': 2.0},
            {'name': 'rsi_period', 'type': int, 'min': 10, 'max': 20, 'default': 14},
        ]
    
    def before(self) -> None:
        """Called before processing each candle"""
        # Initialize variables on first run
        if self.index == 0:
            self.vars['total_trades'] = 0
            self.vars['winning_trades'] = 0
    
    def after(self) -> None:
        """Called after processing each candle"""
        # Update position PnL
        if self.position.is_open:
            self.position.update(self.price)
    
    def on_close_position(self, order) -> None:
        """Called when position closes"""
        self.vars['total_trades'] += 1
        
        if self.position.pnl > 0:
            self.vars['winning_trades'] += 1
            self.log(f"Position closed with PROFIT: ${self.position.pnl:.2f}")
        else:
            self.log(f"Position closed with LOSS: ${self.position.pnl:.2f}")
        
        win_rate = (self.vars['winning_trades'] / self.vars['total_trades']) * 100
        self.log(f"Win Rate: {win_rate:.1f}% ({self.vars['winning_trades']}/{self.vars['total_trades']})")

# Advanced Multi-Indicator Strategy
class AdvancedTrendStrategy(Strategy):
    """
    Advanced strategy using multiple indicators like Jesse examples
    """
    
    def __init__(self, exchange_name: str = 'kraken', symbol: str = 'BTC/USD', timeframe: str = '1h'):
        super().__init__(exchange_name, symbol, timeframe)
        # Strategy parameters
        self.sma_fast = 20
        self.sma_slow = 50
        self.rsi_period = 14
        self.atr_multiplier = 2.0
        self.adx_threshold = 25
    
    @property
    def sma_fast_line(self):
        return ta.sma(self.candles, self.sma_fast)
    
    @property
    def sma_slow_line(self):
        return ta.sma(self.candles, self.sma_slow)
    
    @property
    def rsi_value(self):
        return ta.rsi(self.candles, self.rsi_period)
    
    @property
    def atr_value(self):
        return ta.atr(self.candles)
    
    @property
    def trend_strength(self):
        """Simplified ADX calculation"""
        # In real implementation, use proper ADX
        sma_diff = abs(self.sma_fast_line - self.sma_slow_line)
        return (sma_diff / self.price) * 100
    
    @property
    def big_trend(self):
        """Long-term trend using larger timeframe"""
        # In real implementation, get 4h or daily candles
        if self.sma_fast_line > self.sma_slow_line:
            return 1
        elif self.sma_fast_line < self.sma_slow_line:
            return -1
        return 0
    
    def should_long(self) -> bool:
        return (
            self.big_trend == 1 and
            self.sma_fast_line > self.sma_slow_line and
            self.rsi_value < 70 and
            self.trend_strength > self.adx_threshold
        )
    
    def should_short(self) -> bool:
        return (
            self.big_trend == -1 and
            self.sma_fast_line < self.sma_slow_line and
            self.rsi_value > 30 and
            self.trend_strength > self.adx_threshold
        )
    
    def go_long(self):
        entry = self.price
        stop = entry - (self.atr_value * self.atr_multiplier)
        qty = utils.risk_to_qty(self.available_margin, 3, entry, stop, fee_rate=self.fee_rate)
        self.buy = qty, entry
    
    def go_short(self):
        entry = self.price
        stop = entry + (self.atr_value * self.atr_multiplier)
        qty = utils.risk_to_qty(self.available_margin, 3, entry, stop, fee_rate=self.fee_rate)
        self.sell = qty, entry
    
    def should_cancel_entry(self) -> bool:
        return True
    
    def on_open_position(self, order) -> None:
        if self.is_long:
            self.stop_loss = self.position.qty, self.position.entry_price - (self.atr_value * self.atr_multiplier)
            self.take_profit = self.position.qty, self.position.entry_price + (self.atr_value * 3)
        elif self.is_short:
            self.stop_loss = self.position.qty, self.position.entry_price + (self.atr_value * self.atr_multiplier)
            self.take_profit = self.position.qty, self.position.entry_price - (self.atr_value * 3)
    
    def update_position(self) -> None:
        # Exit on trend change
        if self.is_long and self.big_trend == -1:
            self.liquidate()
        elif self.is_short and self.big_trend == 1:
            self.liquidate()
    
    def hyperparameters(self) -> list:
        return [
            {'name': 'sma_fast', 'type': int, 'min': 10, 'max': 30, 'default': 20},
            {'name': 'sma_slow', 'type': int, 'min': 30, 'max': 100, 'default': 50},
            {'name': 'atr_multiplier', 'type': float, 'min': 1.0, 'max': 4.0, 'default': 2.0},
            {'name': 'adx_threshold', 'type': int, 'min': 15, 'max': 40, 'default': 25},
        ]