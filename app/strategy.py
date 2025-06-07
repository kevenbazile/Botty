# Updated strategy.py - Replace your existing file with this
# File: PS C:\Users\kille\Desktop\bot\app\strategy.py

import os
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Optional

# Add Jesse integration
try:
    from jesse_strategy import ta, utils
    from example_jesse_strategy import GoldenCrossStrategy
    JESSE_AVAILABLE = True
except ImportError:
    JESSE_AVAILABLE = False

# Add Enhanced LLM integration with financial news
try:
    from llm_integration import LLMAnalyzer, NewsFetcher
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Add HFT integration
try:
    from hft_scalping import HFTScalpingStrategy, ScalpingConfig
    HFT_AVAILABLE = True
except ImportError:
    HFT_AVAILABLE = False

# HFT helper functions (simplified versions in case import fails)
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

class TechnicalIndicators:
    """Technical analysis indicators"""
    
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
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD Indicator"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2):
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band

class LiveTradingStrategy:
    def __init__(self):
        # Initialize components
        self.indicators = TechnicalIndicators()
        self.setup_exchange()
        
        # Add HFT Scalping integration
        if HFT_AVAILABLE:
            try:
                self.hft_strategy = HFTScalpingStrategy()
                print("✅ HFT scalping integration enabled")
            except Exception as e:
                print(f"⚠️ HFT initialization failed: {e}")
                self.hft_strategy = None
        else:
            self.hft_strategy = None
            print("⚠️ HFT scalping not available")
        
        # Enhanced LLM integration with financial news
        if LLM_AVAILABLE:
            try:
                self.llm_analyzer = LLMAnalyzer()
                self.news_fetcher = NewsFetcher()
                print("✅ Enhanced LLM + Financial News integration enabled")
            except Exception as e:
                print(f"⚠️ Enhanced LLM initialization failed: {e}")
                self.llm_analyzer = None
                self.news_fetcher = None
        else:
            self.llm_analyzer = None
            self.news_fetcher = None
            print("⚠️ Enhanced LLM + Financial News integration not available")
        
        # Add Jesse strategy if available
        if JESSE_AVAILABLE:
            try:
                self.jesse_strategy = GoldenCrossStrategy('kraken', 'BTC/USD', '1h')
                print("✅ Jesse strategy integration enabled")
            except Exception as e:
                print(f"⚠️ Jesse initialization failed: {e}")
                self.jesse_strategy = None
        else:
            self.jesse_strategy = None
            print("⚠️ Jesse strategy not available")
        
        # Strategy parameters
        self.fast_period = 10
        self.slow_period = 20
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.min_confidence = 60
        
        self.price_history = pd.DataFrame()
    
    def generate_enhanced_signal(self, symbol: str = 'BTC/USD') -> dict:
        """Generate trading signal with financial news + LLM analysis"""
        try:
            df = self.fetch_recent_data(symbol)
            if df.empty:
                return {
                    'signal': 'HOLD', 
                    'reason': 'No data available', 
                    'confidence': 0,
                    'timestamp': datetime.now(),
                    'price': 0
                }
            
            # Technical analysis
            technical_analysis = self.analyze_market_conditions(df)
            
            # Ensure timestamp is included
            if 'timestamp' not in technical_analysis:
                technical_analysis['timestamp'] = datetime.now()
            
            # Jesse analysis if available
            jesse_analysis = None
            if JESSE_AVAILABLE and self.jesse_strategy and 'jesse_ema_fast' in df.columns:
                jesse_analysis = self.get_jesse_analysis(df)
            
            # Enhanced LLM analysis with financial news
            llm_analysis = None
            financial_headlines = []
            
            if self.llm_analyzer and self.news_fetcher:
                try:
                    # Get financial news headlines
                    financial_headlines = self.news_fetcher.get_bitcoin_news(limit=15)
                    print(f"📰 Fetched {len(financial_headlines)} financial news headlines")
                    
                    # Get enhanced analysis including financial news sentiment
                    if financial_headlines:
                        llm_analysis = self.llm_analyzer.analyze_financial_news_sentiment(
                            financial_headlines, technical_analysis
                        )
                        
                        # Get overall news sentiment if available
                        if hasattr(self.news_fetcher, 'enhanced_fetcher') and self.news_fetcher.enhanced_fetcher:
                            news_sentiment = self.news_fetcher.enhanced_fetcher.get_news_sentiment_summary()
                            if news_sentiment:
                                llm_analysis['overall_news_sentiment'] = news_sentiment
                                print(f"📊 Overall News Sentiment: {news_sentiment.get('overall_sentiment', 'Neutral')}")
                    else:
                        # Fallback to regular LLM analysis
                        price_data = {'price': technical_analysis.get('price', 0)}
                        llm_analysis = self.llm_analyzer.analyze_market_sentiment(price_data, technical_analysis)
                        
                except Exception as e:
                    print(f"⚠️ Enhanced LLM + Financial news analysis failed: {e}")
                    llm_analysis = None
            
            # Combine all analyses (technical + jesse + financial news + LLM)
            combined_signal = self.combine_all_signals_enhanced(
                technical_analysis, jesse_analysis, llm_analysis, financial_headlines
            )
            
            # Ensure all required fields are present
            combined_signal.setdefault('timestamp', datetime.now())
            combined_signal.setdefault('price', technical_analysis.get('price', 0))
            combined_signal.setdefault('confidence', 0)
            combined_signal.setdefault('signal', 'HOLD')
            combined_signal.setdefault('reason', 'Enhanced analysis completed')
            
            return combined_signal
            
        except Exception as e:
            print(f"❌ Enhanced signal generation error: {e}")
            return {
                'signal': 'ERROR',
                'reason': f'Enhanced analysis failed: {str(e)}',
                'confidence': 0,
                'timestamp': datetime.now(),
                'price': 0
            }
    
    def combine_all_signals_enhanced(self, technical_analysis: dict, jesse_analysis: dict, 
                                   llm_analysis: dict, financial_headlines: List[str]) -> dict:
        """Enhanced signal combination including financial news sentiment"""
        
        current_price = technical_analysis.get('price', 0)
        
        # Start with enhanced signal structure
        combined = {
            'signal': 'HOLD',
            'reason': 'Multi-factor analysis with financial news',
            'confidence': 0,
            'price': current_price,
            'technical_analysis': technical_analysis,
            'jesse_analysis': jesse_analysis,
            'llm_analysis': llm_analysis,
            'financial_headlines': financial_headlines[:5],  # Include top 5 headlines
            'news_sentiment': 'Neutral',
            'timestamp': datetime.now()
        }
        
        # Calculate weighted confidence scores with financial news
        technical_weight = 0.3   # 30% weight to technical analysis
        jesse_weight = 0.2       # 20% weight to Jesse analysis  
        llm_weight = 0.25        # 25% weight to LLM analysis
        news_weight = 0.25       # 25% weight to financial news sentiment
        
        total_confidence = 0
        signal_votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        # Technical analysis vote
        tech_signal = technical_analysis.get('signal', 'HOLD')
        tech_strength = abs(technical_analysis.get('strength', 0))
        tech_confidence = min(tech_strength * 25, 100)
        
        signal_votes[tech_signal] += technical_weight
        total_confidence += tech_confidence * technical_weight
        
        # Jesse analysis vote
        if jesse_analysis:
            jesse_strength = 0
            if jesse_analysis.get('trend') == 'bullish':
                jesse_signal = 'BUY'
                jesse_strength = 1
            elif jesse_analysis.get('trend') == 'bearish':
                jesse_signal = 'SELL'
                jesse_strength = 1
            else:
                jesse_signal = 'HOLD'
                jesse_strength = 0
            
            signal_votes[jesse_signal] += jesse_weight
            total_confidence += (jesse_strength * 70) * jesse_weight
        
        # LLM analysis vote
        if llm_analysis:
            llm_signal = llm_analysis.get('llm_recommendation', 'HOLD')
            llm_confidence = llm_analysis.get('llm_confidence', 50)
            
            signal_votes[llm_signal] += llm_weight
            total_confidence += llm_confidence * llm_weight
        
        # Financial news sentiment vote (NEW)
        news_signal = 'HOLD'
        news_confidence = 50
        
        if llm_analysis and 'llm_news_sentiment' in llm_analysis:
            news_sentiment = llm_analysis.get('llm_news_sentiment', 'Neutral')
            news_confidence = llm_analysis.get('llm_news_confidence', 50)
            
            combined['news_sentiment'] = news_sentiment
            
            # Convert news sentiment to trading signal
            if news_sentiment == 'Bullish':
                news_signal = 'BUY'
            elif news_sentiment == 'Bearish':
                news_signal = 'SELL'
            
            # Check for key themes
            key_themes = llm_analysis.get('llm_key_themes', [])
            if key_themes:
                combined['key_themes'] = key_themes
                # Boost confidence for institutional themes
                if any('institutional' in theme.lower() for theme in key_themes):
                    news_confidence += 10
            
            signal_votes[news_signal] += news_weight
            total_confidence += news_confidence * news_weight
        
        # Analyze financial headlines directly for additional context
        if financial_headlines:
            bullish_count = 0
            bearish_count = 0
            
            for headline in financial_headlines:
                headline_lower = headline.lower()
                if '📈' in headline:
                    bullish_count += 1
                elif '📉' in headline:
                    bearish_count += 1
        
            # Adjust news sentiment based on emoji analysis
            if bullish_count > bearish_count:
                news_confidence += 5
            elif bearish_count > bullish_count:
                news_confidence -= 5
            
            combined['news_emoji_analysis'] = {
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'total_headlines': len(financial_headlines)
            }
        
        # Source-specific analysis
        if llm_analysis and 'llm_source_analysis' in llm_analysis:
            source_data = llm_analysis['llm_source_analysis']
            combined['source_sentiment'] = source_data
            
            # Weight institutional sources more heavily
            polygon_sentiment = source_data.get('polygon', 'Neutral')
            if polygon_sentiment != 'Neutral':
                total_confidence += 5  # Boost for institutional source
        
        # Overall news sentiment from enhanced fetcher
        if llm_analysis and 'overall_news_sentiment' in llm_analysis:
            overall_sentiment = llm_analysis['overall_news_sentiment']
            combined['overall_news_data'] = overall_sentiment
            
            # Boost confidence if sentiment aligns across sources
            if overall_sentiment.get('confidence', 0) > 60:
                total_confidence += 10
        
        # Determine final signal based on weighted votes
        winning_signal = max(signal_votes, key=signal_votes.get)
        winning_vote = signal_votes[winning_signal]
        
        # Require stronger consensus for buy/sell signals due to news volatility
        consensus_threshold = 0.6  # 60% consensus required
        
        if winning_vote > consensus_threshold:
            combined['signal'] = winning_signal
            combined['confidence'] = min(total_confidence, 100)
            
            # Create detailed reasoning
            reasons = []
            if tech_signal == winning_signal:
                reasons.append(f"Technical: {tech_signal}")
            if jesse_analysis and jesse_analysis.get('trend') == ('bullish' if winning_signal == 'BUY' else 'bearish' if winning_signal == 'SELL' else 'neutral'):
                reasons.append(f"Jesse: {winning_signal}")
            if llm_analysis and llm_analysis.get('llm_recommendation') == winning_signal:
                reasons.append(f"LLM: {winning_signal}")
            if news_signal == winning_signal:
                reasons.append(f"News: {winning_signal}")
            
            combined['reason'] = f"Enhanced consensus {winning_signal}: {', '.join(reasons)}"
            
            # Add financial news context to reason
            if combined['news_sentiment'] != 'Neutral':
                combined['reason'] += f" | News sentiment: {combined['news_sentiment']}"
            
        else:
            # No clear consensus - be conservative
            combined['signal'] = 'HOLD'
            combined['confidence'] = total_confidence * 0.6  # Reduce confidence
            combined['reason'] = "No clear consensus across technical, fundamental, and news signals"
        
        return combined
    
    def setup_exchange(self):
        """Setup exchange connection"""
        try:
            load_dotenv()
            api_key = os.getenv('KRAKEN_API_KEY')
            api_secret = os.getenv('KRAKEN_API_SECRET')
            
            if not api_key or not api_secret:
                print("❌ Missing API credentials in environment variables")
                self.exchange = None
                return
            
            self.exchange = ccxt.kraken({
                'apiKey': api_key,
                'secret': api_secret,
                'sandbox': False,
            })
            print(f"✅ Connected to Kraken exchange")
            
        except Exception as e:
            print(f"❌ Failed to connect to exchange: {e}")
            self.exchange = None
    
    def fetch_recent_data(self, symbol: str = 'BTC/USD', timeframe: str = '1h', limit: int = 100):
        """Fetch recent market data with technical indicators"""
        try:
            if not self.exchange:
                print("❌ Exchange not connected")
                return pd.DataFrame()
                
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add technical indicators
            df = self.add_technical_indicators(df)
            
            # Add Jesse-style indicators if available
            if JESSE_AVAILABLE and self.jesse_strategy:
                df = self.add_jesse_indicators(df)
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return pd.DataFrame()
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators"""
        if len(df) < 50:
            return df
            
        # Moving averages
        df['sma_fast'] = self.indicators.sma(df['close'], self.fast_period)
        df['sma_slow'] = self.indicators.sma(df['close'], self.slow_period)
        df['ema_fast'] = self.indicators.ema(df['close'], self.fast_period)
        df['ema_slow'] = self.indicators.ema(df['close'], self.slow_period)
        
        # RSI
        df['rsi'] = self.indicators.rsi(df['close'], self.rsi_period)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self.indicators.macd(df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.indicators.bollinger_bands(df['close'])
        
        return df
    
    def add_jesse_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Jesse-style indicators"""
        if len(df) < 50:
            return df
        
        try:
            # Simple EMA calculations
            df['jesse_ema_fast'] = df['close'].ewm(span=20).mean()
            df['jesse_ema_slow'] = df['close'].ewm(span=50).mean()
            
            # Simple RSI calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['jesse_rsi'] = 100 - (100 / (1 + rs))
            
            # Simple ATR calculation
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['jesse_atr'] = true_range.rolling(window=14).mean()
            
            return df
            
        except Exception as e:
            print(f"⚠️ Error adding Jesse indicators: {e}")
            # Add safe fallback values
            df['jesse_ema_fast'] = df['close'] 
            df['jesse_ema_slow'] = df['close']
            df['jesse_rsi'] = 50
            df['jesse_atr'] = df['close'] * 0.02
            return df
    
    def analyze_market_conditions(self, df: pd.DataFrame) -> dict:
        """Analyze current market conditions"""
        if df.empty or len(df) < 20:
            return {'trend': 'unknown', 'strength': 0, 'signals': [], 'price': 0}
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        analysis = {
            'price': current['close'],
            'trend': 'unknown',
            'strength': 0,
            'signals': [],
            'indicators': {}
        }
        
        # Trend analysis using moving averages
        if not pd.isna(current['sma_fast']) and not pd.isna(current['sma_slow']):
            if current['sma_fast'] > current['sma_slow']:
                analysis['trend'] = 'bullish'
                analysis['strength'] += 1
            else:
                analysis['trend'] = 'bearish'
                analysis['strength'] -= 1
        
        # RSI analysis
        if not pd.isna(current['rsi']):
            analysis['indicators']['rsi'] = current['rsi']
            if current['rsi'] < self.rsi_oversold:
                analysis['signals'].append('RSI_OVERSOLD')
                analysis['strength'] += 1
            elif current['rsi'] > self.rsi_overbought:
                analysis['signals'].append('RSI_OVERBOUGHT')
                analysis['strength'] -= 1
        
        # MACD analysis
        if not pd.isna(current['macd']) and not pd.isna(current['macd_signal']):
            analysis['indicators']['macd'] = current['macd']
            analysis['indicators']['macd_signal'] = current['macd_signal']
            
            # MACD crossover signals
            if not pd.isna(prev['macd']) and not pd.isna(prev['macd_signal']):
                if current['macd'] > current['macd_signal'] and prev['macd'] <= prev['macd_signal']:
                    analysis['signals'].append('MACD_BULLISH_CROSSOVER')
                    analysis['strength'] += 1
                elif current['macd'] < current['macd_signal'] and prev['macd'] >= prev['macd_signal']:
                    analysis['signals'].append('MACD_BEARISH_CROSSOVER')
                    analysis['strength'] -= 1
        
        # Bollinger Bands analysis
        if not pd.isna(current['bb_upper']) and not pd.isna(current['bb_lower']):
            if current['close'] > current['bb_upper']:
                analysis['signals'].append('BB_UPPER_BREAK')
            elif current['close'] < current['bb_lower']:
                analysis['signals'].append('BB_LOWER_BREAK')
        
        return analysis
    
    def get_jesse_analysis(self, df: pd.DataFrame) -> dict:
        """Get Jesse-style analysis"""
        try:
            current = df.iloc[-1]
            
            # Jesse-style trend analysis
            jesse_trend = 'bullish' if current['jesse_ema_fast'] > current['jesse_ema_slow'] else 'bearish'
            
            # Jesse-style signals
            jesse_signals = []
            if current['jesse_rsi'] < 30:
                jesse_signals.append('JESSE_RSI_OVERSOLD')
            elif current['jesse_rsi'] > 70:
                jesse_signals.append('JESSE_RSI_OVERBOUGHT')
            
            return {
                'trend': jesse_trend,
                'signals': jesse_signals,
                'rsi': current['jesse_rsi'],
                'ema_fast': current['jesse_ema_fast'],
                'ema_slow': current['jesse_ema_slow'],
                'atr': current['jesse_atr']
            }
            
        except Exception as e:
            print(f"⚠️ Jesse analysis error: {e}")
            return {}
    
    def generate_trading_signal(self, symbol: str = 'BTC/USD') -> dict:
        """Generate trading signal based on technical analysis"""
        try:
            df = self.fetch_recent_data(symbol)
            if df.empty:
                return {
                    'signal': 'HOLD', 
                    'reason': 'No data available', 
                    'confidence': 0,
                    'timestamp': datetime.now(),
                    'price': 0
                }
            
            analysis = self.analyze_market_conditions(df)
            current_price = analysis['price']
            
            # Decision logic
            buy_signals = ['RSI_OVERSOLD', 'MACD_BULLISH_CROSSOVER', 'BB_LOWER_BREAK']
            sell_signals = ['RSI_OVERBOUGHT', 'MACD_BEARISH_CROSSOVER', 'BB_UPPER_BREAK']
            
            buy_score = sum(1 for signal in analysis['signals'] if signal in buy_signals)
            sell_score = sum(1 for signal in analysis['signals'] if signal in sell_signals)
            
            # Calculate confidence based on multiple factors
            confidence = min(abs(analysis['strength']) * 25, 100)
            
            signal_data = {
                'signal': 'HOLD',
                'reason': 'No clear signal',
                'confidence': confidence,
                'price': current_price,
                'analysis': analysis,
                'timestamp': datetime.now()
            }
            
            # Buy conditions
            if (buy_score >= 1 and analysis['trend'] == 'bullish') or buy_score >= 2:
                signal_data.update({
                    'signal': 'BUY',
                    'reason': f"Bullish signals: {', '.join([s for s in analysis['signals'] if s in buy_signals])}"
                })
            
            # Sell conditions
            elif (sell_score >= 1 and analysis['trend'] == 'bearish') or sell_score >= 2:
                signal_data.update({
                    'signal': 'SELL',
                    'reason': f"Bearish signals: {', '.join([s for s in analysis['signals'] if s in sell_signals])}"
                })
            
            return signal_data
            
        except Exception as e:
            print(f"❌ Trading signal generation error: {e}")
            return {
                'signal': 'ERROR',
                'reason': f'Analysis failed: {str(e)}',
                'confidence': 0,
                'timestamp': datetime.now(),
                'price': 0
            }

# Enhanced simple_strategy function
def simple_strategy():
    """Enhanced strategy with financial news + LLM analysis"""
    try:
        strategy = LiveTradingStrategy()
        
        if not strategy.exchange:
            print("❌ Cannot run strategy - exchange not connected")
            return
        
        # Use enhanced analysis with financial news
        if strategy.llm_analyzer and strategy.news_fetcher:
            signal_data = strategy.generate_enhanced_signal()
            print("🌐 Using FULL ENHANCED analysis (Technical + Jesse + LLM + Financial News)")
        else:
            # Fall back to standard analysis
            signal_data = strategy.generate_trading_signal()
            print("📊 Using STANDARD analysis")
        
        # Ensure signal_data has required fields
        if not signal_data or 'timestamp' not in signal_data:
            print("❌ Invalid signal data received")
            return
        
        # Enhanced output with financial news information
        timestamp = signal_data.get('timestamp', datetime.now())
        price = signal_data.get('price', 0)
        signal = signal_data.get('signal', 'UNKNOWN')
        confidence = signal_data.get('confidence', 0)
        reason = signal_data.get('reason', 'No reason provided')
        
        print(f"📊 Market Analysis at {timestamp.strftime('%H:%M:%S')}")
        print(f"💰 BTC/USD Price: ${price:,.2f}")
        print(f"📈 Signal: {signal} (Confidence: {confidence:.1f}%)")
        print(f"🔍 Reason: {reason}")
        
        # Technical analysis summary
        if 'technical_analysis' in signal_data and signal_data['technical_analysis']:
            tech = signal_data['technical_analysis']
            print(f"🔧 Technical: {tech.get('trend', 'unknown')} trend | Strength: {tech.get('strength', 0)}")
            if tech.get('signals'):
                print(f"📊 Technical Signals: {', '.join(tech['signals'])}")
        elif 'analysis' in signal_data and signal_data['analysis']:
            # Fallback for standard analysis
            analysis = signal_data['analysis']
            print(f"📊 Trend: {analysis.get('trend', 'unknown')} | Strength: {analysis.get('strength', 0)}")
            if analysis.get('signals'):
                print(f"🚨 Active Signals: {', '.join(analysis['signals'])}")
            
            # Show indicators
            indicators = analysis.get('indicators', {})
            if 'rsi' in indicators:
                rsi_value = indicators['rsi']
                rsi_status = "Oversold" if rsi_value < 30 else "Overbought" if rsi_value > 70 else "Neutral"
                print(f"📈 RSI: {rsi_value:.2f} ({rsi_status})")
                
            if 'macd' in indicators:
                macd_value = indicators['macd']
                macd_signal_val = indicators.get('macd_signal', macd_value)
                macd_trend = "Bullish" if macd_value > macd_signal_val else "Bearish"
                print(f"📈 MACD: {macd_value:.4f} ({macd_trend})")
        
        # Jesse analysis summary
        if 'jesse_analysis' in signal_data and signal_data['jesse_analysis']:
            jesse = signal_data['jesse_analysis']
            print(f"🧠 Jesse: {jesse.get('trend', 'unknown')} trend | RSI: {jesse.get('rsi', 0):.1f}")
            if jesse.get('signals'):
                print(f"🎯 Jesse Signals: {', '.join(jesse['signals'])}")
        
        # LLM analysis summary
        if 'llm_analysis' in signal_data and signal_data['llm_analysis']:
            llm = signal_data['llm_analysis']
            print(f"🤖 LLM: {llm.get('llm_sentiment', 'Neutral')} sentiment | Rec: {llm.get('llm_recommendation', 'HOLD')} | Confidence: {llm.get('llm_confidence', 0)}%")
            
            # Show LLM reasoning (truncated)
            reasoning = llm.get('llm_reasoning', '')
            if reasoning:
                print(f"💭 LLM Reasoning: {reasoning[:150]}{'...' if len(reasoning) > 150 else ''}")
            
            # Show risk level if available
            risk_level = llm.get('llm_risk_level')
            if risk_level:
                print(f"⚠️ LLM Risk Level: {risk_level}")
        
        # NEW: Financial News Analysis Display
        news_sentiment = signal_data.get('news_sentiment', 'Neutral')
        if news_sentiment != 'Neutral':
            print(f"📰 Financial News Sentiment: {news_sentiment}")
        
        # Show financial news headlines
        financial_headlines = signal_data.get('financial_headlines', [])
        if financial_headlines:
            print("📰 Top Financial News Headlines:")
            for i, headline in enumerate(financial_headlines[:3], 1):
                # Truncate long headlines
                display_headline = headline[:100] + '...' if len(headline) > 100 else headline
                print(f"   {i}. {display_headline}")
        
        # Show source sentiment breakdown
        if 'source_sentiment' in signal_data:
            source_data = signal_data['source_sentiment']
            print("📰 News Source Sentiment:")
            for source, sentiment in source_data.items():
                if sentiment != 'Neutral':
                    print(f"   {source.title()}: {sentiment}")
        
        # Show key themes from financial news
        key_themes = signal_data.get('key_themes', [])
        if key_themes:
            print(f"📊 Key Financial Themes: {', '.join(key_themes[:3])}")
        
        # Show overall news sentiment data
        overall_news_data = signal_data.get('overall_news_data')
        if overall_news_data:
            overall_sentiment = overall_news_data.get('overall_sentiment', 'Neutral')
            confidence = overall_news_data.get('confidence', 0)
            article_count = overall_news_data.get('article_count', 0)
            
            if overall_sentiment != 'Neutral':
                print(f"📊 Overall News Analysis: {overall_sentiment} (Confidence: {confidence}%, {article_count} articles)")
        
        # Show news emoji analysis
        emoji_analysis = signal_data.get('news_emoji_analysis')
        if emoji_analysis:
            bullish = emoji_analysis['bullish_count']
            bearish = emoji_analysis['bearish_count']
            total = emoji_analysis['total_headlines']
            
            if bullish > 0 or bearish > 0:
                print(f"📊 News Sentiment Indicators: {bullish} bullish 📈, {bearish} bearish 📉 (from {total} headlines)")
        
        # Enhanced decision logic with financial news context
        if signal == 'BUY':
            if confidence >= 80:
                print("🟢 VERY STRONG BUY SIGNAL - High confidence across all models!")
                if news_sentiment == 'Bullish':
                    print("   📰 Financial news confirms bullish sentiment!")
            elif confidence >= 60:
                print("🟢 STRONG BUY SIGNAL - Good confidence")
                if news_sentiment == 'Bearish':
                    print("   ⚠️ Warning: Financial news sentiment is bearish")
            else:
                print("🟡 WEAK BUY SIGNAL - Low confidence")
        
        elif signal == 'SELL':
            if confidence >= 80:
                print("🔴 VERY STRONG SELL SIGNAL - High confidence across all models!")
                if news_sentiment == 'Bearish':
                    print("   📰 Financial news confirms bearish sentiment!")
            elif confidence >= 60:
                print("🔴 STRONG SELL SIGNAL - Good confidence")
                if news_sentiment == 'Bullish':
                    print("   ⚠️ Warning: Financial news sentiment is bullish")
            else:
                print("🟡 WEAK SELL SIGNAL - Low confidence")
        
        elif signal == 'HOLD':
            if 'No consensus' in reason or 'No clear consensus' in reason:
                print("⚠️ HOLD - Conflicting signals from different models")
                if financial_headlines:
                    print("   📰 Financial news sentiment is mixed")
            else:
                print("⏸️ HOLD - No clear trading opportunity")
        
        elif signal == 'ERROR':
            print(f"❌ ERROR in analysis: {reason}")
        
        # Show financial news fetch status
        if strategy.news_fetcher:
            print("✅ Financial news monitoring active")
        else:
            print("⚠️ Financial news monitoring offline")
        
        print("-" * 80)
        
    except Exception as e:
        print(f"❌ Strategy error: {e}")
        import traceback
        traceback.print_exc()

# Keep backward compatibility
def basic_strategy():
    """Basic strategy for backward compatibility"""
    try:
        load_dotenv()
        
        exchange = ccxt.kraken({
            'apiKey': os.getenv('KRAKEN_API_KEY'),
            'secret': os.getenv('KRAKEN_API_SECRET'),
        })

        symbol = 'BTC/USD'
        ticker = exchange.fetch_ticker(symbol)
        price = ticker['last']
        print(f"📊 Kraken {symbol} price: ${price:,.2f}")

        if price < 65000:
            print("💰 Buy signal on Kraken!")
        else:
            print("❌ Hold off for now.")
            
    except Exception as e:
        print(f"❌ Basic strategy error: {e}")