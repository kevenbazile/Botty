# FIXED: Use simple date format that Alpaca accepts# COMPLETE FIXED news_fetcher.py - REAL API CALLS ONLY
# File: PS C:\Users\kille\Desktop\bot\app\news_fetcher.py

import os
import json
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dotenv import load_dotenv
import re

class FinancialNewsFetcher:
    """Fetch REAL crypto news using your actual API keys - NO SAMPLE DATA"""
    
    def __init__(self):
        load_dotenv()
        
        # Your REAL API keys
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.coindesk_api_key = os.getenv('COINDESK_API_KEY')
        self.alpaca_api_key = os.getenv('ALPACA_API_KEY')
        self.alpaca_secret = os.getenv('ALPACA_SECRET_KEY')
        
        print(f"🔑 News API Key: {'✅ Found' if self.news_api_key else '❌ Missing'}")
        print(f"🔑 Alpaca API Key: {'✅ Found' if self.alpaca_api_key else '❌ Missing'}")
        
        # Real API endpoints
        self.newsapi_base_url = "https://files.polygon.io"
        self.alpaca_base_url = "https://data.alpaca.markets/v1beta1/news"
        
        # Crypto keywords for filtering
        self.crypto_keywords = [
            'bitcoin', 'btc', 'crypto', 'cryptocurrency', 'ethereum', 'eth',
            'altcoin', 'blockchain', 'defi', 'web3', 'digital currency',
            'coinbase', 'binance', 'kraken', 'bull market', 'bear market',
            'hodl', 'satoshi', 'whale', 'mining', 'halving'
        ]
        
        # Cache
        self.content_cache = set()
        
    def fetch_newsapi_crypto(self, max_results: int = 10) -> List[Dict]:
        """Fetch REAL crypto news from NewsAPI.org using your API key"""
        if not self.news_api_key:
            print("❌ NewsAPI key missing")
            return []
        
        crypto_news = []
        
        try:
            url = f"{self.newsapi_base_url}/everything"
            params = {
                'q': 'bitcoin OR cryptocurrency OR crypto OR ethereum',
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': max_results,
                'from': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                'apiKey': self.news_api_key
            }
            
            print(f"📡 Calling NewsAPI...")
            response = requests.get(url, params=params, timeout=15)
            print(f"📡 NewsAPI Response: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                print(f"📰 NewsAPI returned {len(articles)} articles")
                
                for article in articles:
                    title = article.get('title', '')
                    description = article.get('description', '')
                    source_name = article.get('source', {}).get('name', 'NewsAPI')
                    
                    if self.contains_crypto_keywords(title + ' ' + description):
                        crypto_news.append({
                            'source': f'NewsAPI-{source_name}',
                            'title': self.clean_text(title),
                            'description': self.clean_text(description or '')[:300],
                            'author': source_name,
                            'published_at': article.get('publishedAt', ''),
                            'url': article.get('url', ''),
                            'sentiment_score': self._analyze_simple_sentiment(title + ' ' + description)
                        })
            
            else:
                print(f"❌ NewsAPI Error: {response.status_code}")
        
        except Exception as e:
            print(f"❌ NewsAPI fetch error: {e}")
        
        return crypto_news
    
    def fetch_alpaca_crypto_news(self, max_results: int = 10) -> List[Dict]:
        """Fetch REAL crypto news from Alpaca - FIXED DATE FORMAT"""
        if not self.alpaca_api_key or not self.alpaca_secret:
            print("❌ Alpaca keys missing")
            return []
        
        crypto_news = []
        
        try:
            headers = {
                'ALPCA-API-KEY-ID': self.alpaca_api_key,
                'ALPCA-API-SECRET-KEY': self.alpaca_secret
            }
            
            start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            params = {
                'symbols': 'BTCUSD',
                'start': start_date,
                'sort': 'desc',
                'page_size': min(max_results, 10)
            }
            
            print(f"📡 Calling Alpaca API with date: {start_date}")
            response = requests.get(self.alpaca_base_url, headers=headers, params=params, timeout=15)
            print(f"📡 Alpaca Response: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('news', [])
                print(f"📰 Alpaca returned {len(articles)} articles")
                
                for article in articles:
                    headline = article.get('headline', '')
                    summary = article.get('summary', '')
                    
                    if self.contains_crypto_keywords(headline + ' ' + summary):
                        crypto_news.append({
                            'source': 'Alpaca',
                            'title': self.clean_text(headline),
                            'description': self.clean_text(summary or '')[:300],
                            'author': 'Alpaca News',
                            'published_at': article.get('created_at', ''),
                            'url': article.get('url', ''),
                            'sentiment_score': self._analyze_simple_sentiment(headline + ' ' + summary)
                        })
            
            else:
                print(f"❌ Alpaca Error: {response.status_code} - {response.text}")
        
        except Exception as e:
            print(f"❌ Alpaca fetch error: {e}")
        
        return crypto_news
    
    def fetch_coindesk_rss(self, max_results: int = 5) -> List[Dict]:
        """Fetch REAL news from CoinDesk RSS feed"""
        crypto_news = []
        
        try:
            import feedparser
            
            print("📡 Calling CoinDesk RSS feed")
            rss_url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
            
            feed = feedparser.parse(rss_url)
            
            if feed.entries:
                print(f"📰 CoinDesk RSS returned {len(feed.entries)} articles")
                
                for entry in feed.entries[:max_results]:
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    
                    crypto_news.append({
                        'source': 'CoinDesk',
                        'title': self.clean_text(title),
                        'description': self.clean_text(summary or '')[:300],
                        'author': 'CoinDesk',
                        'published_at': entry.get('published', ''),
                        'url': entry.get('link', ''),
                        'sentiment_score': self._analyze_simple_sentiment(title + ' ' + summary)
                    })
            else:
                print("❌ No articles from CoinDesk RSS")
            
        except ImportError:
            print("❌ feedparser not installed. Install with: pip install feedparser")
        except Exception as e:
            print(f"❌ CoinDesk RSS fetch error: {e}")
        
        return crypto_news
    
    def contains_crypto_keywords(self, text: str) -> bool:
        """Check if text contains crypto-related keywords"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.crypto_keywords)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\!\?\,\-\$\%\#]', '', text)
        return text.strip()[:200]
    
    def _analyze_simple_sentiment(self, text: str) -> float:
        """Simple sentiment analysis using keyword matching"""
        text_lower = text.lower()
        
        positive_keywords = [
            'bull', 'bullish', 'rise', 'up', 'gain', 'profit', 'positive', 
            'moon', 'pump', 'green', 'surge', 'rally', 'breakthrough', 'adoption',
            'all-time high', 'record', 'increase', 'growth', 'strong'
        ]
        
        negative_keywords = [
            'bear', 'bearish', 'fall', 'down', 'loss', 'negative', 'crash',
            'dump', 'red', 'decline', 'drop', 'sell-off', 'correction', 'concern'
        ]
        
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return 0.0
        
        sentiment_score = (positive_count - negative_count) / total_sentiment_words
        return max(-1.0, min(1.0, sentiment_score))
    
    def get_all_crypto_news(self, max_per_source: int = 5) -> List[str]:
        """Get REAL crypto news from all your APIs - FIXED VERSION"""
        print("📰 Fetching REAL crypto news from your APIs...")
        
        all_news = []
        headlines = []
        
        # Fetch from NewsAPI
        try:
            newsapi_news = self.fetch_newsapi_crypto(max_per_source)
            all_news.extend(newsapi_news)
        except Exception as e:
            print(f"⚠️ NewsAPI error: {e}")
        
        # Fetch from Alpaca (fixed)
        try:
            alpaca_news = self.fetch_alpaca_crypto_news(max_per_source)
            all_news.extend(alpaca_news)
        except Exception as e:
            print(f"⚠️ Alpaca error: {e}")
        
        # Fetch from CoinDesk
        try:
            coindesk_news = self.fetch_coindesk_free(max_per_source)
            all_news.extend(coindesk_news)
        except Exception as e:
            print(f"⚠️ CoinDesk error: {e}")
        
        print(f"📊 Total articles fetched: {len(all_news)}")
        
        # Convert to headlines with sentiment indicators
        for article in all_news:
            source = article['source']
            title = article['title']
            sentiment_score = article.get('sentiment_score', 0)
            
            if sentiment_score > 0.2:
                sentiment_emoji = " 📈"
            elif sentiment_score < -0.2:
                sentiment_emoji = " 📉"
            else:
                sentiment_emoji = " ➡️"
            
            headline = f"[{source}] {title}{sentiment_emoji}"
            headlines.append(headline)
        
        # Save raw data
        self.save_news_data(all_news)
        
        if headlines:
            print(f"✅ Fetched {len(headlines)} REAL crypto news headlines")
            return headlines[:15]
        else:
            print("⚠️ No real news fetched, using minimal fallback")
            return [
                "[System] Financial news APIs temporarily unavailable ➡️",
                "[System] Using technical analysis for trading decisions ➡️"
            ]
    
    def save_news_data(self, articles: List[Dict]):
        """Save news data for analysis"""
        try:
            data_file = os.path.join(os.path.dirname(__file__), 'news_data.json')
            
            existing_data = []
            if os.path.exists(data_file):
                with open(data_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            
            for article in articles:
                article['fetched_at'] = datetime.now().isoformat()
                existing_data.append(article)
            
            existing_data = existing_data[-500:]
            
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            print(f"⚠️ Error saving news data: {e}")
    
    def get_news_sentiment_summary(self) -> Dict:
        """Get overall news sentiment summary"""
        try:
            data_file = os.path.join(os.path.dirname(__file__), 'news_data.json')
            
            if not os.path.exists(data_file):
                return {'overall_sentiment': 'Neutral', 'confidence': 60}
            
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
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
                return {'overall_sentiment': 'Neutral', 'confidence': 50}
            
            sentiment_scores = [article.get('sentiment_score', 0) for article in recent_articles]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            
            if avg_sentiment > 0.1:
                overall_sentiment = 'Bullish'
            elif avg_sentiment < -0.1:
                overall_sentiment = 'Bearish'
            else:
                overall_sentiment = 'Neutral'
            
            confidence = min(len(recent_articles) * 15 + 50, 100)
            
            return {
                'overall_sentiment': overall_sentiment,
                'confidence': confidence,
                'avg_sentiment_score': avg_sentiment,
                'article_count': len(recent_articles),
                'sources': list(set(article.get('source', 'Unknown') for article in recent_articles))
            }
            
        except Exception as e:
            print(f"⚠️ Error analyzing sentiment: {e}")
            return {'overall_sentiment': 'Neutral', 'confidence': 50}

# Enhanced NewsFetcher class for backward compatibility
class EnhancedNewsFetcher:
    """Enhanced news fetcher using REAL APIs - FIXED VERSION"""
    
    def __init__(self):
        self.financial_fetcher = FinancialNewsFetcher()
    
    def get_bitcoin_news(self, limit: int = 15) -> List[str]:
        """Get Bitcoin news from REAL financial sources"""
        return self.financial_fetcher.get_all_crypto_news(max_per_source=5)[:limit]
    
    def get_news_sentiment_summary(self) -> Dict:
        """Get current news sentiment analysis"""
        return self.financial_fetcher.get_news_sentiment_summary()

# For backward compatibility
NewsFetcher = EnhancedNewsFetcher

if __name__ == "__main__":
    # Test the FIXED news fetcher
    fetcher = FinancialNewsFetcher()
    headlines = fetcher.get_all_crypto_news()
    
    print("\n📰 REAL Crypto Headlines:")
    for i, headline in enumerate(headlines, 1):
        print(f"{i}. {headline}")
    
    sentiment = fetcher.get_news_sentiment_summary()
    print(f"\n📊 Overall Sentiment: {sentiment}")
