class LiveTradingExecutor:
    """Handles actual trade execution - ADD THIS TO YOUR EXISTING CODE"""
    
    def __init__(self, exchange, enable_trading=True):
        self.exchange = exchange
        self.enable_trading = enable_trading
        self.last_trade_time = None
        self.min_trade_interval = 300  # 5 minutes between trades
        self.position_size = 0.1  # 10% of balance per trade
        self.min_confidence = 75  # Minimum confidence to trade
        self.trade_log = []
        
    def can_trade_now(self):
        """Check if we can trade (time and system checks)"""
        if not self.enable_trading:
            return False, "Live trading disabled"
            
        if not self.exchange:
            return False, "Exchange not connected"
            
        # Time-based trading limits
        if self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
            if time_since_last < self.min_trade_interval:
                return False, f"Too soon since last trade ({time_since_last:.0f}s ago)"
        
        return True, "Ready to trade"
    
    def get_trading_balance(self):
        """Get current trading balance"""
        try:
            balance = self.exchange.fetch_balance()
            usd_free = balance.get('USD', {}).get('free', 0)
            btc_free = balance.get('BTC', {}).get('free', 0)
            
            return {
                'USD': usd_free,
                'BTC': btc_free,
                'total_usd_value': usd_free  # Could add BTC conversion
            }
        except Exception as e:
            print(f"❌ Error fetching balance: {e}")
            return {'USD': 0, 'BTC': 0, 'total_usd_value': 0}
    
    def execute_buy_order(self, symbol, current_price, confidence, reason):
        """Execute actual BUY order on exchange"""
        try:
            # Get available balance
            balance = self.get_trading_balance()
            usd_available = balance['USD']
            
            if usd_available < 25:  # Minimum $25 to trade
                print(f"❌ Insufficient USD balance: ${usd_available:.2f}")
                return False
            
            # Calculate BTC quantity to buy
            trade_amount = usd_available * self.position_size
            btc_quantity = trade_amount / current_price
            btc_quantity = round(btc_quantity, 8)  # Round to 8 decimals
            
            # Check minimum trade size (Kraken minimum is usually 0.0001 BTC)
            if btc_quantity < 0.0001:
                print(f"❌ Trade size too small: {btc_quantity} BTC")
                return False
            
            print(f"\n🟢 EXECUTING LIVE BUY ORDER")
            print(f"   💰 USD Available: ${usd_available:.2f}")
            print(f"   📊 Trade Amount: ${trade_amount:.2f} ({self.position_size*100}%)")
            print(f"   ₿ BTC Quantity: {btc_quantity:.8f}")
            print(f"   💡 Confidence: {confidence}%")
            print(f"   📝 Reason: {reason}")
            
            # PLACE ACTUAL ORDER
            order = self.exchange.create_market_buy_order(symbol, btc_quantity)
            
            print(f"✅ BUY ORDER EXECUTED!")
            print(f"   🆔 Order ID: {order['id']}")
            print(f"   📊 Status: {order['status']}")
            print(f"   ₿ Amount: {order.get('amount', btc_quantity)} BTC")
            
            # Log the trade
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'action': 'BUY',
                'symbol': symbol,
                'quantity': btc_quantity,
                'price': current_price,
                'confidence': confidence,
                'reason': reason,
                'order_id': order['id'],
                'status': 'executed'
            }
            
            self.trade_log.append(trade_record)
            self.last_trade_time = datetime.now()
            
            return True
            
        except Exception as e:
            print(f"❌ BUY ORDER FAILED: {e}")
            return False
    
    def execute_sell_order(self, symbol, current_price, confidence, reason):
        """Execute actual SELL order on exchange"""
        try:
            # Get available BTC balance
            balance = self.get_trading_balance()
            btc_available = balance['BTC']
            
            if btc_available < 0.0001:  # Minimum BTC to sell
                print(f"❌ Insufficient BTC balance: {btc_available:.8f}")
                return False
            
            # Sell all available BTC (minus small buffer for fees)
            btc_quantity = round(btc_available * 0.99, 8)  # 99% to account for fees
            estimated_usd = btc_quantity * current_price
            
            print(f"\n🔴 EXECUTING LIVE SELL ORDER")
            print(f"   ₿ BTC Available: {btc_available:.8f}")
            print(f"   ₿ Selling: {btc_quantity:.8f} BTC")
            print(f"   💰 Estimated USD: ${estimated_usd:.2f}")
            print(f"   💡 Confidence: {confidence}%")
            print(f"   📝 Reason: {reason}")
            
            # PLACE ACTUAL ORDER
            order = self.exchange.create_market_sell_order(symbol, btc_quantity)
            
            print(f"✅ SELL ORDER EXECUTED!")
            print(f"   🆔 Order ID: {order['id']}")
            print(f"   📊 Status: {order['status']}")
            print(f"   💰 Estimated Value: ${estimated_usd:.2f}")
            
            # Log the trade
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'action': 'SELL',
                'symbol': symbol,
                'quantity': btc_quantity,
                'price': current_price,
                'confidence': confidence,
                'reason': reason,
                'order_id': order['id'],
                'status': 'executed'
            }
            
            self.trade_log.append(trade_record)
            self.last_trade_time = datetime.now()
            
            return True
            
        except Exception as e:
            print(f"❌ SELL ORDER FAILED: {e}")
            return False

# MODIFY YOUR EXISTING LiveTradingStrategy class
# Add this to your __init__ method:

def __init__(self):
    # ... your existing __init__ code ...
    
    # ADD THESE LINES to enable live trading:
    self.live_trading_enabled = True  # SET THIS TO TRUE
    
    if self.exchange and self.live_trading_enabled:
        self.trading_executor = LiveTradingExecutor(self.exchange, enable_trading=True)
        print("🚀 LIVE TRADING EXECUTOR ENABLED!")
        print("⚠️  WARNING: Bot will place REAL orders with REAL money!")
    else:
        self.trading_executor = None
        print("📊 Paper trading mode only")

# REPLACE YOUR CURRENT simple_strategy() function with this:

def simple_strategy():
    """Enhanced strategy with AUTOMATIC LIVE TRADING"""
    try:
        strategy = LiveTradingStrategy()
        
        if not strategy.exchange:
            print("❌ Cannot run strategy - exchange not connected")
            return
        
        # Generate enhanced signal with all your existing analysis
        if strategy.llm_analyzer and strategy.news_fetcher:
            signal_data = strategy.generate_enhanced_signal()
            print("🌐 Using FULL ENHANCED analysis (Technical + Jesse + LLM + Financial News)")
        else:
            signal_data = strategy.generate_trading_signal()
            print("📊 Using STANDARD analysis")
        
        # Display all your existing analysis output
        timestamp = signal_data.get('timestamp', datetime.now())
        price = signal_data.get('price', 0)
        signal = signal_data.get('signal', 'UNKNOWN')
        confidence = signal_data.get('confidence', 0)
        reason = signal_data.get('reason', 'No reason provided')
        
        print(f"📊 Market Analysis at {timestamp.strftime('%H:%M:%S')}")
        print(f"💰 BTC/USD Price: ${price:,.2f}")
        print(f"📈 Signal: {signal} (Confidence: {confidence:.1f}%)")
        print(f"🔍 Reason: {reason}")
        
        # ... ALL your existing analysis display code stays the same ...
        
        # NEW: AUTOMATIC TRADE EXECUTION
        if strategy.trading_executor:
            can_trade, trade_status = strategy.trading_executor.can_trade_now()
            
            print(f"\n🚀 LIVE TRADING STATUS: {trade_status}")
            
            if can_trade and confidence >= strategy.trading_executor.min_confidence:
                
                if signal == 'BUY':
                    print(f"🟢 EXECUTING AUTOMATIC BUY ORDER...")
                    success = strategy.trading_executor.execute_buy_order(
                        'BTC/USD', price, confidence, reason
                    )
                    if success:
                        print("✅ LIVE BUY ORDER COMPLETED!")
                    
                elif signal == 'SELL':
                    print(f"🔴 EXECUTING AUTOMATIC SELL ORDER...")
                    success = strategy.trading_executor.execute_sell_order(
                        'BTC/USD', price, confidence, reason
                    )
                    if success:
                        print("✅ LIVE SELL ORDER COMPLETED!")
                
                else:
                    print("⏸️ Signal is HOLD - no trade executed")
            
            elif confidence < strategy.trading_executor.min_confidence:
                print(f"⚠️ Confidence {confidence}% below minimum {strategy.trading_executor.min_confidence}% - no trade")
            
            else:
                print(f"⏰ Cannot trade: {trade_status}")
            
            # Show current trading status
            balance = strategy.trading_executor.get_trading_balance()
            print(f"\n💰 Current Balance:")
            print(f"   USD: ${balance['USD']:.2f}")
            print(f"   BTC: {balance['BTC']:.8f}")
            print(f"   Total Trades: {len(strategy.trading_executor.trade_log)}")
            
        else:
            print("📊 PAPER TRADING MODE - Analysis only, no real trades")
        
        print("-" * 80)
        
    except Exception as e:
        print(f"❌ Strategy error: {e}")
        import traceback
        traceback.print_exc()
