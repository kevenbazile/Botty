import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import ccxt
from dataclasses import dataclass

@dataclass
class ScalpingConfig:
    """Configuration for HFT scalping strategy"""
    # Profit targets (accounting for fees)
    min_profit_bps: int = 15  # 0.15% minimum profit (15 basis points)
    max_profit_bps: int = 50  # 0.50% maximum profit target
    
    # Risk management
    stop_loss_bps: int = 10   # 0.10% stop loss
    max_position_size: float = 0.02  # 2% of balance max
    
    # Trading parameters
    fee_rate: float = 0.001   # 0.1% fee (Kraken taker fee)
    slippage_bps: int = 5     # 0.05% expected slippage
    
    # Timing
    min_hold_seconds: int = 30     # Minimum hold time (30 seconds)
    max_hold_seconds: int = 300    # Maximum hold time (5 minutes)
    cooldown_seconds: int = 60     # Cooldown between trades
    
    # Entry conditions
    volatility_threshold: float = 0.002  # 0.2% minimum volatility
    volume_multiplier: float = 1.5       # 1.5x average volume required

class HFTScalpingStrategy:
    """High-Frequency Trading scalping strategy for Bitcoin"""
    
    def __init__(self, config: ScalpingConfig = None):
        self.config = config or ScalpingConfig()
        self.position = None
        self.last_trade_time = None
        self.trade_history = []
        self.balance = 10000.0  # Starting balance
        self.daily_trades = 0
        self.daily_profit = 0.0
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_fees_paid = 0.0
        
        # Market data cache
        self.price_cache = []
        self.volume_cache = []
        
    def calculate_profit_target(self, entry_price: float, direction: str) -> Dict:
        """Calculate profit targets accounting for fees and slippage"""
        
        # Total cost per trade (fees + slippage)
        total_cost_bps = (self.config.fee_rate * 2 * 10000) + self.config.slippage_bps
        
        # Minimum profit needed to cover costs + minimum profit
        min_profit_needed = total_cost_bps + self.config.min_profit_bps
        
        if direction == 'LONG':
            stop_loss = entry_price * (1 - self.config.stop_loss_bps / 10000)
            min_target = entry_price * (1 + min_profit_needed / 10000)
            max_target = entry_price * (1 + self.config.max_profit_bps / 10000)
        else:  # SHORT
            stop_loss = entry_price * (1 + self.config.stop_loss_bps / 10000)
            min_target = entry_price * (1 - min_profit_needed / 10000)
            max_target = entry_price * (1 - self.config.max_profit_bps / 10000)
        
        return {
            'stop_loss': stop_loss,
            'min_target': min_target,
            'max_target': max_target,
            'breakeven': entry_price,
            'total_cost_bps': total_cost_bps
        }
    
    def calculate_position_size(self, price: float, risk_amount: float = None) -> float:
        """Calculate position size based on risk management"""
        if risk_amount is None:
            risk_amount = self.balance * self.config.max_position_size
        
        # Account for fees in position sizing
        fee_buffer = 1 + (self.config.fee_rate * 2)  # Entry + exit fees
        max_position_value = risk_amount / fee_buffer
        
        return max_position_value / price
    
    def detect_scalping_opportunity(self, market_data: Dict) -> Optional[Dict]:
        """Detect short-term scalping opportunities"""
        
        # Check cooldown period
        if self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
            if time_since_last < self.config.cooldown_seconds:
                return None
        
        # Check if position already open
        if self.position:
            return None
        
        current_price = market_data['price']
        volatility = market_data.get('volatility', 0)
        volume_ratio = market_data.get('volume_ratio', 1)
        rsi = market_data.get('rsi', 50)
        
        # Minimum volatility check
        if volatility < self.config.volatility_threshold:
            return None
        
        # Volume check
        if volume_ratio < self.config.volume_multiplier:
            return None
        
        # Price momentum signals for scalping
        price_trend = market_data.get('short_trend', 'neutral')
        
        opportunity = None
        
        # LONG scalping opportunity
        if (rsi < 45 and price_trend == 'bullish' and 
            self._check_support_level(current_price, market_data)):
            
            targets = self.calculate_profit_target(current_price, 'LONG')
            opportunity = {
                'direction': 'LONG',
                'entry_price': current_price,
                'size': self.calculate_position_size(current_price),
                'targets': targets,
                'confidence': self._calculate_confidence(market_data, 'LONG'),
                'reason': f'Long scalp: RSI {rsi:.1f}, vol {volatility:.3f}, trend {price_trend}'
            }
        
        # SHORT scalping opportunity  
        elif (rsi > 55 and price_trend == 'bearish' and 
              self._check_resistance_level(current_price, market_data)):
            
            targets = self.calculate_profit_target(current_price, 'SHORT')
            opportunity = {
                'direction': 'SHORT',
                'entry_price': current_price,
                'size': self.calculate_position_size(current_price),
                'targets': targets,
                'confidence': self._calculate_confidence(market_data, 'SHORT'),
                'reason': f'Short scalp: RSI {rsi:.1f}, vol {volatility:.3f}, trend {price_trend}'
            }
        
        return opportunity
    
    def _check_support_level(self, price: float, market_data: Dict) -> bool:
        """Check if price is near support for long entry"""
        # Simple support check - price bouncing off recent low
        recent_low = market_data.get('recent_low', price)
        return abs(price - recent_low) / recent_low < 0.001  # Within 0.1% of recent low
    
    def _check_resistance_level(self, price: float, market_data: Dict) -> bool:
        """Check if price is near resistance for short entry"""
        # Simple resistance check - price hitting recent high
        recent_high = market_data.get('recent_high', price)
        return abs(price - recent_high) / recent_high < 0.001  # Within 0.1% of recent high
    
    def _calculate_confidence(self, market_data: Dict, direction: str) -> float:
        """Calculate confidence score for the trade"""
        confidence = 50.0
        
        volatility = market_data.get('volatility', 0)
        volume_ratio = market_data.get('volume_ratio', 1)
        rsi = market_data.get('rsi', 50)
        
        # Higher volatility = higher confidence (up to a point)
        if 0.002 <= volatility <= 0.005:
            confidence += 20
        elif volatility > 0.005:
            confidence += 10  # Too much volatility is risky
        
        # Volume confirmation
        if volume_ratio > 2:
            confidence += 15
        elif volume_ratio > 1.5:
            confidence += 10
        
        # RSI extremes for scalping
        if direction == 'LONG' and rsi < 40:
            confidence += 15
        elif direction == 'SHORT' and rsi > 60:
            confidence += 15
        
        return min(confidence, 100)
    
    def execute_scalping_trade(self, opportunity: Dict) -> Dict:
        """Execute a scalping trade"""
        
        entry_time = datetime.now()
        direction = opportunity['direction']
        entry_price = opportunity['entry_price']
        size = opportunity['size']
        targets = opportunity['targets']
        
        # Calculate fees
        entry_fee = size * entry_price * self.config.fee_rate
        
        # Create position
        self.position = {
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': entry_time,
            'size': size,
            'targets': targets,
            'entry_fee': entry_fee,
            'opportunity': opportunity
        }
        
        self.total_trades += 1
        self.daily_trades += 1
        
        return {
            'action': 'ENTER',
            'direction': direction,
            'price': entry_price,
            'size': size,
            'fee': entry_fee,
            'targets': targets,
            'timestamp': entry_time
        }
    
    def manage_position(self, current_price: float, current_time: datetime = None) -> Optional[Dict]:
        """Manage existing position - check for exit conditions"""
        
        if not self.position:
            return None
        
        if current_time is None:
            current_time = datetime.now()
        
        direction = self.position['direction']
        entry_price = self.position['entry_price']
        entry_time = self.position['entry_time']
        size = self.position['size']
        targets = self.position['targets']
        
        hold_time = (current_time - entry_time).total_seconds()
        
        # Time-based exit (maximum hold time)
        if hold_time > self.config.max_hold_seconds:
            return self._close_position(current_price, 'TIME_EXIT', current_time)
        
        # Don't exit too quickly
        if hold_time < self.config.min_hold_seconds:
            return None
        
        # Profit target hit
        if direction == 'LONG':
            if current_price >= targets['min_target']:
                return self._close_position(current_price, 'PROFIT_TARGET', current_time)
            elif current_price <= targets['stop_loss']:
                return self._close_position(current_price, 'STOP_LOSS', current_time)
        
        else:  # SHORT
            if current_price <= targets['min_target']:
                return self._close_position(current_price, 'PROFIT_TARGET', current_time)
            elif current_price >= targets['stop_loss']:
                return self._close_position(current_price, 'STOP_LOSS', current_time)
        
        return None
    
    def _close_position(self, exit_price: float, reason: str, exit_time: datetime) -> Dict:
        """Close the current position"""
        
        if not self.position:
            return None
        
        direction = self.position['direction']
        entry_price = self.position['entry_price']
        entry_time = self.position['entry_time']
        size = self.position['size']
        entry_fee = self.position['entry_fee']
        
        # Calculate exit fee
        exit_fee = size * exit_price * self.config.fee_rate
        total_fees = entry_fee + exit_fee
        
        # Calculate P&L
        if direction == 'LONG':
            gross_pnl = (exit_price - entry_price) * size
        else:  # SHORT
            gross_pnl = (entry_price - exit_price) * size
        
        net_pnl = gross_pnl - total_fees
        pnl_percentage = (net_pnl / (entry_price * size)) * 100
        
        hold_time = (exit_time - entry_time).total_seconds()
        
        # Update statistics
        self.total_fees_paid += total_fees
        self.daily_profit += net_pnl
        self.balance += net_pnl
        
        if net_pnl > 0:
            self.winning_trades += 1
        
        # Record trade
        trade_record = {
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'size': size,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'pnl_percentage': pnl_percentage,
            'total_fees': total_fees,
            'hold_time_seconds': hold_time,
            'exit_reason': reason
        }
        
        self.trade_history.append(trade_record)
        self.last_trade_time = exit_time
        
        # Clear position
        self.position = None
        
        return {
            'action': 'EXIT',
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'net_pnl': net_pnl,
            'pnl_percentage': pnl_percentage,
            'fees': total_fees,
            'hold_time': hold_time,
            'reason': reason,
            'timestamp': exit_time
        }
    
    def get_daily_stats(self) -> Dict:
        """Get daily trading statistics"""
        if not self.trade_history:
            return {'message': 'No trades today'}
        
        today_trades = [t for t in self.trade_history 
                       if t['entry_time'].date() == datetime.now().date()]
        
        if not today_trades:
            return {'message': 'No trades today'}
        
        winning_trades = len([t for t in today_trades if t['net_pnl'] > 0])
        total_pnl = sum(t['net_pnl'] for t in today_trades)
        total_fees = sum(t['total_fees'] for t in today_trades)
        avg_hold_time = np.mean([t['hold_time_seconds'] for t in today_trades])
        
        return {
            'total_trades': len(today_trades),
            'winning_trades': winning_trades,
            'win_rate': (winning_trades / len(today_trades)) * 100,
            'total_pnl': total_pnl,
            'total_fees': total_fees,
            'avg_hold_time': avg_hold_time,
            'trades_per_hour': len(today_trades) / max(1, (datetime.now().hour + 1)),
            'current_balance': self.balance
        }
    
    def should_continue_trading(self) -> bool:
        """Check if should continue trading based on daily limits and performance"""
        
        # Daily trade limit
        if self.daily_trades >= 50:  # Max 50 trades per day
            return False
        
        # Daily loss limit
        if self.daily_profit < -100:  # Stop if daily loss > $100
            return False
        
        # Balance protection
        if self.balance < self.config.max_position_size * 1000:  # Keep minimum balance
            return False
        
        return True

# Market data helper functions
def calculate_volatility(prices: List[float], periods: int = 20) -> float:
    """Calculate recent price volatility"""
    if len(prices) < periods:
        return 0
    
    recent_prices = prices[-periods:]
    returns = [((recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]) 
               for i in range(1, len(recent_prices))]
    
    return np.std(returns) if returns else 0

def detect_short_trend(prices: List[float], periods: int = 10) -> str:
    """Detect very short-term trend for scalping"""
    if len(prices) < periods:
        return 'neutral'
    
    recent_prices = prices[-periods:]
    slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
    
    threshold = recent_prices[-1] * 0.0005  # 0.05% threshold
    
    if slope > threshold:
        return 'bullish'
    elif slope < -threshold:
        return 'bearish'
    else:
        return 'neutral'

def calculate_volume_ratio(volumes: List[float], periods: int = 20) -> float:
    """Calculate current volume vs average volume"""
    if len(volumes) < periods + 1:
        return 1.0
    
    current_volume = volumes[-1]
    avg_volume = np.mean(volumes[-periods-1:-1])
    
    return current_volume / avg_volume if avg_volume > 0 else 1.0