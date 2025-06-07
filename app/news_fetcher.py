# COMPLETELY FIXED news_fetcher.py - REAL API CALLS ONLY
# File: app/news_fetcher.py

import os
import json
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dotenv import load_dotenv
import re

class RealFinancialNewsFetcher:
    """Fetch REAL crypto news using your actual API keys - NO FAKE DATA"""
    
    def __init__(self):
        load_dotenv()
        
        # Your REAL API keys
        self.alpaca_api_key = os.getenv('ALPACA_API_KEY')
        self.alpaca_secret = os.getenv('ALPACA_SECRET_KEY')
        self.polygon_api_key = os.getenv('POLYGON_API_KEY')
        self.coindesk_api_key = os.getenv('COINDESK_API_KEY')
        
        print(f"🔑 API Keys Status:")
        print(f"   Alpaca: {'✅ Found' if self.alpaca_api_key else '❌ Missing'}")
        print(f"   Polygon: {'✅ Found' if self.polygon_api_key else '❌ Missing'}")
        print(f"   CoinDesk: {'✅ Found' if self.coindesk_api_key else '❌ Missing'}")
        
        # Real API endpoints
        self.alpaca_base_url = "https://data.alpaca.markets/v1beta1/news"
        self.polygon_base_url = "https://api.polygon.io/v2/reference/news"
        self.coindesk_rss_url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
        
        # Crypto keywords for filtering
        self.crypto_keywords = [
            'bitcoin', 'btc', 'crypto', 'cryptocurrency', 'ethereum', 'eth',
            'altcoin', 'blockchain', 'defi', 'web3', 'digital currency',
            'coinbase', 'binance', 'kraken', 'bull market', 'bear market',
            'hodl', 'satoshi', 'whale', 'mining', 'halving', 'metaverse',
            'nft', 'solana', 'cardano', 'polkadot', 'chainlink', 'dogecoin'
        ]
        
        # NO SAMPLE DATA - ONLY REAL API CALLS
        
    def fetch_alpaca_crypto_news(self, max_results: int = 10) -> List[Dict]:
        """Fetch REAL crypto news from Alpaca Markets API"""
        if not self.alpaca_api_key or not self.alpaca_secret:
            print("❌ Alpaca API credentials missing")
            return []
        
        crypto_news = []
        
        try:
            headers = {
                'APCA-API-KEY-ID': self.alpaca_api_key,
                'APCA-API-SECRET-KEY': self.alpaca_secret
            }
            
            # FIXED: Use proper date format for Alpaca
            start_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%dT%H:%M:%S')
            
            # Try different symbol approaches for crypto news
            crypto_symbols = ['BTCUSD', 'ETHUSD', 'BTC', 'ETH']
            
            for symbol in crypto_symbols:
                try:
                    params = {
                        'symbols': symbol,
                        'start': start_date,
                        'sort': 'desc',
                        'include_content': 'true',
                        'exclude_contentless': 'true'
                    }
                    
                    print(f"📡 Calling Alpaca API for {symbol}...")
                    response = requests.get(
                        self.alpaca_base_url, 
                        headers=headers, 
                        params=params, 
                        timeout=15
                    )
                    
                    print(f"📡 Alpaca Response ({symbol}): {response.status_code}")
                    
                    if response.status_code == 200:
                        data = response.json()
                        articles = data.get('news', [])
                        print(f"📰 Alpaca returned {len(articles)} articles for {symbol}")
                        
                        for article in articles:
                            headline = article.get('headline', '')
                            summary = article.get('summary', '')
                            content = article.get('content', '')
                            
                            # Check if crypto-related
                            full_text = f"{headline} {summary} {content}".lower()
                            if self.contains_crypto_keywords(full_text):
                                crypto_news.append({
                                    'source': 'Alpaca',
                                    'title': self.clean_text(headline),
                                    'description': self.clean_text(summary or content[:300]),
                                    'author': article.get('author', 'Alpaca News'),
                                    'published_at': article.get('created_at', ''),
                                    'url': article.get('url', ''),
                                    'symbol': symbol,
                                    'sentiment_score': self._analyze_simple_sentiment(full_text)
                                })
                        
                        # If we got results, don't need to try other symbols
                        if articles:
                            break
                            
                    elif response.status_code == 422:
                        print(f"⚠️ Alpaca: Invalid parameters for {symbol}")
                        continue
                    else:
                        print(f"❌ Alpaca Error ({symbol}): {response.status_code} - {response.text[:200]}")
                        
                except Exception as symbol_error:
                    print(f"❌ Alpaca error for {symbol}: {symbol_error}")
                    continue
                    
        except Exception as e:
            print(f"❌ Alpaca fetch error: {e}")
        
        return crypto_news[:max_results]
    
    def fetch_polygon_crypto_news(self, max_results: int = 10) -> List[Dict]:
        """Fetch REAL crypto news from Polygon.io API"""
        if not self.polygon_api_key:
            print("❌ Polygon API key missing")
            return []
        
        crypto_news = []
        
        try:
            # Polygon.io news endpoint with crypto tickers
            params = {
                'ticker': 'X:BTCUSD',  # Polygon crypto ticker format
                'published_utc.gte': (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d'),
                'order': 'desc',
                'limit': max_results * 2,  # Get more to filter for crypto
                'apikey': self.polygon_api_key
            }
            
            print(f"📡 Calling Polygon.io API...")
            response = requests.get(
                self.polygon_base_url, 
                params=params, 
                timeout=15
            )
            
            print(f"📡 Polygon Response: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('results', [])
                print(f"📰 Polygon returned {len(articles)} articles")
                
                for article in articles:
                    title = article.get('title', '')
                    description = article.get('description', '')
                    
                    # Check if crypto-related
                    full_text = f"{title} {description}".lower()
                    if self.contains_crypto_keywords(full_text):
                        crypto_news.append({
                            'source': 'Polygon.io',
                            'title': self.clean_text(title),
                            'description': self.clean_text(description or '')[:300],
                            'author': article.get('author', 'Polygon News'),
                            'published_at': article.get('published_utc', ''),
                            'url': article.get('article_url', ''),
                            'sentiment_score': self._analyze_simple_sentiment(full_text)
                        })
            
            else:
                print(f"❌ Polygon Error: {response.status_code} - {response.text[:200]}")
                
        except Exception as e:
            print(f"❌ Polygon fetch error: {e}")
        
        return crypto_news[:max_results]
    
    def fetch_coindesk_rss_news(self, max_results: int = 10) -> List[Dict]:
        """Fetch REAL news from CoinDesk RSS feed"""
        crypto_news = []
        
        try:
            import feedparser
            
            print(f"📡 Calling CoinDesk RSS feed...")
            
            # CoinDesk RSS feed
            response = requests.get(self.coindesk_rss_url, timeout=15)
            
            if response.status_code == 200:
                feed = feedparser.parse(response.content)
                print(f"📰 CoinDesk RSS returned {len(feed.entries)} articles")
                
                for entry in feed.entries[:max_results]:
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    
                    crypto_news.append({
                        'source': 'CoinDesk',
                        'title': self.clean_text(title),
                        'description': self.clean_text(summary)[:300],
                        'author': 'CoinDesk',
                        'published_at': entry.get('published', ''),
                        'url': entry.get('link', ''),
                        'sentiment_score': self._analyze_simple_sentiment(f"{title} {summary}")
                    })
            else:
                print(f"❌ CoinDesk RSS Error: {response.status_code}")
                
        except ImportError:
            print("❌ feedparser not installed. Run: pip install feedparser")
        except Exception as e:
            print(f"❌ CoinDesk RSS fetch error: {e}")
        
        return crypto_news
    
    def fetch_free_crypto_news_apis(self, max_results: int = 5) -> List[Dict]:
        """Fetch from free crypto news APIs as backup"""
        crypto_news = []
        
        try:
            # CryptoPanic free API (no key required for basic use)
            cryptopanic_url = "https://cryptopanic.com/api/v1/posts/"
            params = {
                'auth_token': 'free',  # Free tier
                'public': 'true',
                'currencies': 'BTC,ETH',
                'filter': 'hot',
                'regions': 'en'
            }
            
            print("📡 Calling CryptoPanic free API...")
            response = requests.get(cryptopanic_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                posts = data.get('results', [])
                print(f"📰 CryptoPanic returned {len(posts)} posts")
                
                for post in posts[:max_results]:
                    title = post.get('title', '')
                    
                    crypto_news.append({
                        'source': 'CryptoPanic',
                        'title': self.clean_text(title),
                        'description': '',
                        'author': 'CryptoPanic',
                        'published_at': post.get('published_at', ''),
                        'url': post.get('url', ''),
                        'sentiment_score': self._analyze_simple_sentiment(title)
                    })
            else:
                print(f"❌ CryptoPanic Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Free API fetch error: {e}")
        
        return crypto_news
    
    def contains_crypto_keywords(self, text: str) -> bool:
        """Check if text contains crypto-related keywords"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.crypto_keywords)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\!\?\,\-\$\%\#]', '', text)
        return text.strip()[:200]
    
    def _analyze_simple_sentiment(self, text: str) -> float:
        """Simple sentiment analysis using keyword matching"""
        if not text:
            return 0.0
            
        text_lower = text.lower()
        
        positive_keywords = [
            'bull', 'bullish', 'rise', 'up', 'gain', 'profit', 'positive', 
            'moon', 'pump', 'green', 'surge', 'rally', 'breakthrough', 'adoption',
            'all-time high', 'record', 'increase', 'growth', 'strong', 'soar',
            'rocket', 'explode', 'skyrocket', 'breakout', 'momentum'
        ]
        
        negative_keywords = [
            'bear', 'bearish', 'fall', 'down', 'loss', 'negative', 'crash',
            'dump', 'red', 'decline', 'drop', 'sell-off', 'correction', 'concern',
            'plunge', 'collapse', 'fear', 'panic', 'volatile', 'risk'
        ]
        
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return 0.0
        
        sentiment_score = (positive_count - negative_count) / total_sentiment_words
        return max(-1.0, min(1.0, sentiment_score))
    
    def get_all_crypto_news_real(self, max_per_source: int = 5) -> List[str]:
        """Get REAL crypto news from all available APIs - NO FAKE DATA"""
        print("📰 Fetching REAL crypto news from APIs...")
        
        all_news = []
        headlines = []
        
        # 1. Fetch from Alpaca (your primary source)
        try:
            alpaca_news = self.fetch_alpaca_crypto_news(max_per_source)
            all_news.extend(alpaca_news)
            print(f"✅ Alpaca: {len(alpaca_news)} crypto articles")
        except Exception as e:
            print(f"⚠️ Alpaca error: {e}")
        
        # 2. Fetch from Polygon.io
        try:
            polygon_news = self.fetch_polygon_crypto_news(max_per_source)
            all_news.extend(polygon_news)
            print(f"✅ Polygon: {len(polygon_news)} crypto articles")
        except Exception as e:
            print(f"⚠️ Polygon error: {e}")
        
        # 3. Fetch from CoinDesk RSS
        try:
            coindesk_news = self.fetch_coindesk_rss_news(max_per_source)
            all_news.extend(coindesk_news)
            print(f"✅ CoinDesk: {len(coindesk_news)} crypto articles")
        except Exception as e:
            print(f"⚠️ CoinDesk error: {e}")
        
        # 4. Fetch from free APIs as backup
        if len(all_news) < 5:  # Only if we don't have enough real news
            try:
                free_news = self.fetch_free_crypto_news_apis(max_per_source)
                all_news.extend(free_news)
                print(f"✅ Free APIs: {len(free_news)} crypto articles")
            except Exception as e:
                print(f"⚠️ Free API error: {e}")
        
        print(f"📊 Total REAL articles fetched: {len(all_news)}")
        
        # Convert to headlines with sentiment indicators
        for article in all_news:
            source = article['source']
            title = article['title']
            sentiment_score = article.get('sentiment_score', 0)
            
            # Add sentiment emoji based on analysis
            if sentiment_score > 0.2:
                sentiment_emoji = " 📈"
            elif sentiment_score < -0.2:
                sentiment_emoji = " 📉"
            else:
                sentiment_emoji = " ➡️"
            
            headline = f"[{source}] {title}{sentiment_emoji}"
            headlines.append(headline)
        
        # Save raw data for analysis
        self.save_news_data(all_news)
        
        if headlines:
            print(f"✅ Successfully fetched {len(headlines)} REAL crypto news headlines")
            return headlines[:15]  # Return top 15
        else:
            print("❌ No real news fetched from any API")
            # ONLY return this if APIs are truly down
            return [
                "[System] All financial news APIs temporarily unavailable ➡️",
                "[System] Check API credentials and network connection ➡️"
            ]
    
    def save_news_data(self, articles: List[Dict]):
        """Save news data for analysis"""
        if not articles:
            return
            
        try:
            data_file = os.path.join(os.path.dirname(__file__), 'real_news_data.json')
            
            # Load existing data
            existing_data = []
            if os.path.exists(data_file):
                with open(data_file, 'r', encoding='utf-8') as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        existing_data = []
            
            # Add new articles with timestamp
            for article in articles:
                article['fetched_at'] = datetime.now().isoformat()
                existing_data.append(article)
            
            # Keep only last 1000 articles
            existing_data = existing_data[-1000:]
            
            # Save back to file
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved {len(articles)} articles to {data_file}")
            
        except Exception as e:
            print(f"⚠️ Error saving news data: {e}")
    
    def get_news_sentiment_summary(self) -> Dict:
        """Get overall news sentiment summary from REAL data"""
        try:
            data_file = os.path.join(os.path.dirname(__file__), 'real_news_data.json')
            
            if not os.path.exists(data_file):
                return {'overall_sentiment': 'Neutral', 'confidence': 50, 'source': 'No data'}
            
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Only analyze recent articles (last 6 hours)
            recent_cutoff = datetime.now() - timedelta(hours=6)
            recent_articles = []
            
            for article in data:
                try:
                    fetched_time = datetime.fromisoformat(article.get('fetched_at', ''))
                    if fetched_time > recent_cutoff:
                        recent_articles.append(article)
                except:
                    continue
            
            if not recent_articles:
                return {'overall_sentiment': 'Neutral', 'confidence': 50, 'source': 'No recent data'}
            
            # Calculate sentiment
            sentiment_scores = [article.get('sentiment_score', 0) for article in recent_articles]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            
            if avg_sentiment > 0.1:
                overall_sentiment = 'Bullish'
            elif avg_sentiment < -0.1:
                overall_sentiment = 'Bearish'
            else:
                overall_sentiment = 'Neutral'
            
            # Calculate confidence based on article count and sentiment consistency
            confidence = min(len(recent_articles) * 10 + 50, 95)
            
            # Get source breakdown
            sources = list(set(article.get('source', 'Unknown') for article in recent_articles))
            
            return {
                'overall_sentiment': overall_sentiment,
                'confidence': confidence,
                'avg_sentiment_score': avg_sentiment,
                'article_count': len(recent_articles),
                'sources': sources,
                'source': 'Real API data'
            }
            
        except Exception as e:
            print(f"⚠️ Error analyzing sentiment: {e}")
            return {'overall_sentiment': 'Neutral', 'confidence': 50, 'source': 'Error'}

# Enhanced NewsFetcher class for backward compatibility
class EnhancedNewsFetcher:
    """Enhanced news fetcher using REAL APIs ONLY"""
    
    def __init__(self):
        self.real_fetcher = RealFinancialNewsFetcher()
    
    def get_bitcoin_news(self, limit: int = 15) -> List[str]:
        """Get Bitcoin news from REAL financial sources ONLY"""
        return self.real_fetcher.get_all_crypto_news_real(max_per_source=5)[:limit]
    
    def get_news_sentiment_summary(self) -> Dict:
        """Get current news sentiment analysis from REAL data"""
        return self.real_fetcher.get_news_sentiment_summary()

# For backward compatibility
NewsFetcher = EnhancedNewsFetcher

# TEST FUNCTION
def test_real_news_fetcher():
    """Test the real news fetcher with your APIs"""
    print("🧪 Testing REAL News Fetcher...")
    
    fetcher = RealFinancialNewsFetcher()
    
    # Test each API individually
    print("\n1. Testing Alpaca API...")
    alpaca_news = fetcher.fetch_alpaca_crypto_news(3)
    for news in alpaca_news:
        print(f"   📰 {news['title'][:100]}...")
    
    print("\n2. Testing Polygon API...")
    polygon_news = fetcher.fetch_polygon_crypto_news(3)
    for news in polygon_news:
        print(f"   📰 {news['title'][:100]}...")
    
    print("\n3. Testing CoinDesk RSS...")
    coindesk_news = fetcher.fetch_coindesk_rss_news(3)
    for news in coindesk_news:
        print(f"   📰 {news['title'][:100]}...")
    
    print("\n4. Testing Free APIs...")
    free_news = fetcher.fetch_free_crypto_news_apis(3)
    for news in free_news:
        print(f"   📰 {news['title'][:100]}...")
    
    print("\n5. Testing Combined Fetcher...")
    all_headlines = fetcher.get_all_crypto_news_real()
    print(f"\n📊 Final Result: {len(all_headlines)} real headlines:")
    for i, headline in enumerate(all_headlines, 1):
        print(f"   {i}. {headline}")
    
    print(f"\n📈 Sentiment Summary:")
    sentiment = fetcher.get_news_sentiment_summary()
    print(f"   {sentiment}")

if __name__ == "__main__":
    # Run the test
    test_real_news_fetcher()
