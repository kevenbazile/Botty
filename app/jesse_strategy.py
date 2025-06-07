import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Union, List, Dict, Tuple, Optional
import ccxt

# Jesse-style indicators (simplified versions)
class Indicators:
    @staticmethod
    def sma(candles: np.ndarray, period: int = 20, source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
        """Simple Moving Average"""
        if source_type == "close":
            prices = candles[:, 2] if len(candles.shape) > 1 else candles
        elif source_type == "open":
            prices = candles[:, 1] if len(candles.shape) > 1 else candles
        elif source_type == "high":
            prices = candles[:, 3] if len(candles.shape) > 1 else candles
        elif source_type == "low":
            prices = candles[:, 4] if len(candles.shape) > 1 else candles
        else:
            prices = candles[:, 2] if len(candles.shape) > 1 else candles
            
        sma_values = pd.Series(prices).rolling(window=period).mean().values
        
        if sequential:
            return sma_values
        return sma_values[-1] if not np.isnan(sma_values[-1]) else 0

    @staticmethod
    def ema(candles: np.ndarray, period: int = 20, source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
        """Exponential Moving Average"""
        if source_type == "close":
            prices = candles[:, 2] if len(candles.shape) > 1 else candles
        else:
            prices = candles[:, 2] if len(candles.shape) > 1 else candles
            
        ema_values = pd.Series(prices).ewm(span=period).mean().values
        
        if sequential:
            return ema_values
        return ema_values[-1] if not np.isnan(ema_values[-1]) else 0

    @staticmethod
    def rsi(candles: np.ndarray, period: int = 14, source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
        """Relative Strength Index"""
        if source_type == "close":
            prices = candles[:, 2] if len(candles.shape) > 1 else candles
        else:
            prices = candles[:, 2] if len(candles.shape) > 1 else candles
            
        delta = pd.Series(prices).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi_values = (100 - (100 / (1 + rs))).values
        
        if sequential:
            return rsi_values
        return rsi_values[-1] if not np.isnan(rsi_values[-1]) else 50

    @staticmethod
    def atr(candles: np.ndarray, period: int = 14, sequential: bool = False) -> Union[float, np.ndarray]:
        """Average True Range"""
        if len(candles.shape) == 1:
            return 0
            
        high = candles[:, 3]
        low = candles[:, 4]
        close = candles[:, 2]
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        atr_values = pd.Series(tr).rolling(window=period).mean().values
        
        if sequential:
            return atr_values
        return atr_values[-1] if not np.isnan(atr_values[-1]) else 0

# Jesse-style Position class
class Position:
    def __init__(self):
        self.entry_price: float = 0
        self.qty: float = 0
        self.opened_at: float = 0
        self.value: float = 0
        self.type: str = 'close'  # 'long', 'short', 'close'
        self.pnl: float = 0
        self.pnl_percentage: float = 0
        self.is_open: bool = False

    def update(self, current_price: float):
        if self.is_open and self.qty != 0:
            if self.type == 'long':
                self.pnl = (current_price - self.entry_price) * self.qty
            elif self.type == 'short':
                self.pnl = (self.entry_price - current_price) * self.qty
            
            self.pnl_percentage = (self.pnl / (self.entry_price * abs(self.qty))) * 100 if self.entry_price != 0 else 0

# Jesse-style Utils
class Utils:
    @staticmethod
    def size_to_qty(position_size: float, price: float, precision: int = 3, fee_rate: float = 0) -> float:
        """Convert position size to quantity"""
        return round(position_size / price, precision)
    
    @staticmethod
    def qty_to_size(qty: float, price: float) -> float:
        """Convert quantity to position size"""
        return qty * price
    
    @staticmethod
    def risk_to_qty(capital: float, risk_percentage: float, entry_price: float, 
                   stop_loss_price: float, precision: int = 3, fee_rate: float = 0) -> float:
        """Calculate position quantity based on risk percentage"""
        risk_amount = capital * (risk_percentage / 100)
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit == 0:
            return 0
            
        qty = risk_amount / risk_per_unit
        return round(qty, precision)

# Base Strategy Class (Jesse-compatible)
class Strategy(ABC):
    def __init__(self, exchange_name: str = 'kraken', symbol: str = 'BTC/USD', timeframe: str = '1h'):
        self.exchange_name = exchange_name
        self.symbol = symbol
        self.timeframe = timeframe
        self.candles = np.array([])
        self.current_candle = np.array([])
        self.position = Position()
        self.balance = 10000.0
        self.available_margin = 10000.0
        self.index = 0
        self.vars = {}
        self.shared_vars = {}
        
        # Jesse-style order management
        self.buy = None  # (qty, price)
        self.sell = None  # (qty, price)
        self.stop_loss = None  # (qty, price)
        self.take_profit = None  # (qty, price)
        
        # Trading state
        self.is_backtesting = True
        self.is_livetrading = False
        self.leverage = 1
        self.fee_rate = 0.001
        
        # Performance tracking
        self.trades = []
        self.orders = []
        
    # Jesse-style properties
    @property
    def price(self) -> float:
        """Current price (close price)"""
        return self.current_candle[2] if len(self.current_candle) > 0 else 0
    
    @property
    def close(self) -> float:
        """Alias for price"""
        return self.price
    
    @property
    def open(self) -> float:
        """Current candle open price"""
        return self.current_candle[1] if len(self.current_candle) > 0 else 0
    
    @property
    def high(self) -> float:
        """Current candle high price"""
        return self.current_candle[3] if len(self.current_candle) > 0 else 0
    
    @property
    def low(self) -> float:
        """Current candle low price"""
        return self.current_candle[4] if len(self.current_candle) > 0 else 0
    
    @property
    def is_open(self) -> bool:
        """Check if position is open"""
        return self.position.is_open
    
    @property
    def is_close(self) -> bool:
        """Check if position is closed"""
        return not self.position.is_open
    
    @property
    def is_long(self) -> bool:
        """Check if current position is long"""
        return self.position.type == 'long' and self.position.is_open
    
    @property
    def is_short(self) -> bool:
        """Check if current position is short"""
        return self.position.type == 'short' and self.position.is_open

    # Abstract methods (must be implemented by strategy)
    @abstractmethod
    def should_long(self) -> bool:
        """Determine if should enter long position"""
        pass
    
    @abstractmethod
    def should_short(self) -> bool:
        """Determine if should enter short position"""
        pass
    
    @abstractmethod
    def go_long(self):
        """Execute long entry logic"""
        pass
    
    @abstractmethod
    def go_short(self):
        """Execute short entry logic"""
        pass
    
    # Optional methods (can be overridden)
    def should_cancel_entry(self) -> bool:
        """Determine if should cancel pending orders"""
        return False
    
    def update_position(self) -> None:
        """Update existing position (trailing stops, exits, etc.)"""
        pass
    
    def before(self) -> None:
        """Called before each candle processing"""
        pass
    
    def after(self) -> None:
        """Called after each candle processing"""
        pass
    
    def on_open_position(self, order) -> None:
        """Called when position is opened"""
        pass
    
    def on_close_position(self, order) -> None:
        """Called when position is closed"""
        pass
    
    def filters(self) -> List:
        """Return list of filter functions"""
        return []
    
    def hyperparameters(self) -> List[Dict]:
        """Define hyperparameters for optimization"""
        return []
    
    # Utility methods
    def liquidate(self):
        """Close position immediately"""
        if self.position.is_open:
            self.position.is_open = False
            self.position.type = 'close'
            self.position.qty = 0
    
    def log(self, msg: str, log_type: str = 'info'):
        """Log message"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {msg}")
    
    def get_candles(self, exchange: str, symbol: str, timeframe: str) -> np.ndarray:
        """Get candles for different timeframe/symbol"""
        # In a real implementation, this would fetch actual data
        # For now, return current candles
        return self.candles

# Indicators alias (Jesse-style)
ta = Indicators()
utils = Utils()