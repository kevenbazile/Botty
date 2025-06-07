import os
import json
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dotenv import load_dotenv
import re

class FinancialNewsFetcher:
    """Fetch REAL crypto news using your actual API keys"""
    
    def __init__(self):
        load_dotenv()
        
        # Your REAL API keys
        self.news_api_key = os.getenv('NEWS_API_KEY')  # ec8866f4c0ce1237a2f9c8ceb521cab4581152d2501936ee5a0b0234aeece9d7
        self.coindesk_api_key = os.getenv('COINDESK_API_KEY')  # bvIGCfhCRAXB0IfPi47MeVGHf0tZoz9a
        self.alpaca_api_key = os.getenv('ALPACA_API_KEY')  # AKHRJHUNISETI2RVYA93
        self.alpaca_secret = os.getenv('ALPACA_SECRET_KEY')  # t7fHjwEMznRmq0aYpJJW29kwBTMQV2QrqkSinS4b
        
        print(f"🔑 News API Key: {'✅ Found' if self.news_api_key else '❌ Missing'}")
        print(f"🔑 Alpaca API Key: {'✅ Found' if self.alpaca_api_key else '❌ Missing'}")
        
        # Real API endpoints
        self.newsapi_base_url = "https://newsapi.org/v2"
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
            # Real NewsAPI.org request
            url = f"{self.newsapi_base_url}/everything"
            params = {
                'q': 'bitcoin OR cryptocurrency OR crypto OR ethereum OR "digital currency"',
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': max_results,
                'from': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                'apiKey': self.news_api_key
            }
            
            print(f"📡 Calling NewsAPI with key ending in: ...{self.news_api_key[-6:]}")
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
                    published_at = article.get('publishedAt', '')
                    url_link = article.get('url', '')
                    
                    # Filter for crypto content
                    if self.contains_crypto_keywords(title + ' ' + description):
                        content_hash = hash(title + source_name)
                        if content_hash not in self.content_cache:
                            self.content_cache.add(content_hash)
                            
                            crypto_news.append({
                                'source': f'NewsAPI-{source_name}',
                                'title': self.clean_text(title),
                                'description': self.clean_text(description or '')[:300],
                                'author': source_name,
                                'published_at': published_at,
                                'url': url_link,
                                'sentiment_score': self._analyze_simple_sentiment(title + ' ' + description)
                            })
            
            elif response.status_code == 426:
                print("❌ NewsAPI: Upgrade Required - Free tier exhausted")
            elif response.status_code == 401:
                print("❌ NewsAPI: Invalid API key")
            else:
                print(f"❌ NewsAPI Error: {response.status_code} - {response.text}")
        
        except Exception as e:
            print(f"❌ NewsAPI fetch error: {e}")
        
        return crypto_news
    
    def fetch_alpaca_crypto_news(self, max_results: int = 10) -> List[Dict]:
        """Fetch REAL crypto news from Alpaca using your API keys"""
        if not self.alpaca_api_key or not self.alpaca_secret:
            print("❌ Alpaca keys missing")
            return []
        
        crypto_news = []
        
        try:
            headers = {
                'APCA-API-KEY-ID': self.alpaca_api_key,
                'APCA-API-SECRET-KEY': self.alpaca_secret
            }
            
            # Real Alpaca news request
            params = {
                'symbols': 'BTCUSD,ETHUSD,CRYPTO',
                'start': (datetime.now() - timedelta(days=1)).isoformat(),
                'end': datetime.now().isoformat(),
                'sort': 'desc',
                'include_content': 'true',
                'exclude_contentless': 'true',
                'page_size': max_results
            }
            
            print(f"📡 Calling Alpaca API with key ending in: ...{self.alpaca_api_key[-4:]}")
            response = requests.get(self.alpaca_base_url, headers=headers, params=params, timeout=15)
            print(f"📡 Alpaca Response: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('news', [])
                print(f"📰 Alpaca returned {len(articles)} articles")
                
                for article in articles:
                    headline = article.get('headline', '')
                    summary = article.get('summary', '')
                    author = article.get('author', 'Alpaca News')
                    created_at = article.get('created_at', '')
                    url_link = article.get('url', '')
                    
                    if self.contains_crypto_keywords(headline + ' ' + summary):
                        content_hash = hash(headline + author)
                        if content_hash not in self.content_cache:
                            self.content_cache.add(content_hash)
                            
                            crypto_news.append({
                                'source': 'Alpaca',
                                'title': self.clean_text(headline),
                                'description': self.clean_text(summary or '')[:300],
                                'author': author,
                                'published_at': created_at,
                                'url': url_link,
                                'sentiment_score': self._analyze_simple_sentiment(headline + ' ' + summary)
                            })
            
            elif response.status_code == 401:
                print("❌ Alpaca: Invalid API credentials")
            elif response.status_code == 403:
                print("❌ Alpaca: Access forbidden - check account permissions")
            else:
                print(f"❌ Alpaca Error: {response.status_code} - {response.text}")
        
        except Exception as e:
            print(f"❌ Alpaca fetch error: {e}")
        
        return crypto_news
    
    def fetch_coindesk_free(self, max_results: int = 5) -> List[Dict]:
        """Fetch news from CoinDesk free RSS"""
        crypto_news = []
        
        try:
            # CoinDesk RSS feed (free)
            rss_url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
            
            print("📡 Calling CoinDesk RSS feed")
            response = requests.get(rss_url, timeout=10)
            
            if response.status_code == 200:
                # Parse RSS manually (simple approach)
                content = response.text
                
                # Extract titles from RSS (basic parsing)
                import re
                titles = re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', content)
                descriptions = re.findall(r'<description><!\[CDATA\[(.*?)\]\]></description>', content)
                
                for i, title in enumerate(titles[:max_results]):
                    description = descriptions[i] if i < len(descriptions) else ''
                    
                    crypto_news.append({
                        'source': 'CoinDesk',
                        'title': self.clean_text(title),
                        'description': self.clean_text(description)[:300],
                        'author': 'CoinDesk',
                        'published_at': datetime.now().isoformat(),
                        'url': 'https://coindesk.com',
                        'sentiment_score': self._analyze_simple_sentiment(title + ' ' + description)
                    })
                
                print(f"📰 CoinDesk returned {len(crypto_news)} articles")
            
        except Exception as e:
            print(f"❌ CoinDesk fetch error: {e}")
        
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
            'all-time high', 'record', 'increase', 'growth', 'strong', 'breakthrough'
        ]
        
        negative_keywords = [
            'bear', 'bearish', 'fall', 'down', 'loss', 'negative', 'crash',
            'dump', 'red', 'decline', 'drop', 'sell-off', 'correction', 'concern',
            'weakness', 'volatility', 'risk', 'uncertainty', 'fear'
        ]
        
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return 0.0
        
        sentiment_score = (positive_count - negative_count) / total_sentiment_words
        return max(-1.0, min(1.0, sentiment_score))
    
    def get_all_crypto_news(self, max_per_source: int = 5) -> List[str]:
        """Get REAL crypto news from all your APIs"""
        print("📰 Fetching REAL crypto news from your APIs...")
        
        all_news = []
        headlines = []
        
        # Fetch from all REAL sources
        newsapi_news = self.fetch_newsapi_crypto(max_per_source)
        alpaca_news = self.fetch_alpaca_crypto_news(max_per_source)
        coindesk_news = self.fetch_coindesk_free(max_per_source)
        
        all_news.extend(newsapi_news)
        all_news.extend(alpaca_news)
        all_news.extend(coindesk_news)
        
        print(f"📊 Total articles fetched: {len(all_news)}")
        
        # Convert to headlines with sentiment indicators
        for article in all_news:
            source = article['source']
            title = article['title']
            sentiment_score = article.get('sentiment_score', 0)
            
            # Add sentiment indicator
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
            return self._minimal_fallback()
    
    def _minimal_fallback(self) -> List[str]:
        """Minimal fallback if APIs fail"""
        return [
            "[System] Fetching real news from APIs... ➡️",
            "[System] Crypto market analysis in progress ➡️"
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
    """Enhanced news fetcher using REAL APIs"""
    
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
    # Test the REAL news fetcher
    fetcher = FinancialNewsFetcher()
    headlines = fetcher.get_all_crypto_news()
    
    print("\n📰 REAL Crypto Headlines:")
    for i, headline in enumerate(headlines, 1):
        print(f"{i}. {headline}")
    
    sentiment = fetcher.get_news_sentiment_summary()
    print(f"\n📊 Overall Sentiment: {sentiment}")
