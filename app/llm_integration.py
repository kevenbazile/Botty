import os
import json
import requests
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Import the enhanced news fetcher
try:
    from news_fetcher import EnhancedNewsFetcher
    NEWS_FETCHER_AVAILABLE = True
except ImportError:
    NEWS_FETCHER_AVAILABLE = False
    print("⚠️ Enhanced news fetcher not available")

class LLMAnalyzer:
    """LLM-powered market analysis using OpenRouter with financial news integration"""
    
    def __init__(self):
        # FORCE RELOAD THE .ENV FILE
        load_dotenv(override=True)
        
        # TRY MULTIPLE WAYS TO GET THE API KEY
        self.api_key = None
        
        # Method 1: Direct environment variable
        self.api_key = os.getenv('OPEN_ROUTER_KEY')
        
        # Method 2: Try alternative names
        if not self.api_key:
            self.api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            self.api_key = os.getenv('OPENROUTER_KEY')
        
        # Method 3: Try from os.environ directly
        if not self.api_key:
            self.api_key = os.environ.get('OPEN_ROUTER_KEY')
        
        # Clean the API key
        if self.api_key:
            self.api_key = self.api_key.strip().strip('"').strip("'")
        
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Fixed model name - using valid OpenRouter model
        self.model = "meta-llama/llama-3.1-8b-instruct:free"
        
        if not self.api_key:
            print("❌ OPEN_ROUTER_KEY not found in environment variables")
            print("❌ Make sure your .env file has: OPEN_ROUTER_KEY=sk-or-v1-...")
    
    def analyze_market_sentiment(self, price_data: Dict, technical_analysis: Dict) -> Dict:
        """Get LLM analysis of market conditions"""
        if not self.api_key:
            return {"analysis": "LLM not available - missing API key", "confidence": 0}
        
        try:
            # Prepare market data for LLM
            prompt = self._create_market_analysis_prompt(price_data, technical_analysis)
            
            # Get LLM response
            response = self._call_llm(prompt)
            
            # Parse LLM response
            analysis = self._parse_llm_response(response)
            
            return analysis
            
        except Exception as e:
            print(f"❌ LLM analysis error: {e}")
            return {"analysis": f"LLM error: {str(e)}", "confidence": 0}
    
    def analyze_financial_news_sentiment(self, news_headlines: List[str], technical_analysis: Dict) -> Dict:
        """Enhanced analysis using financial news headlines"""
        if not self.api_key:
            return {"analysis": "Financial news analysis not available - missing API key", "confidence": 0}
            
        if not news_headlines:
            return {"analysis": "No financial news headlines provided", "confidence": 0}
        
        try:
            # Create enhanced prompt with financial news data
            prompt = self._create_financial_news_analysis_prompt(news_headlines, technical_analysis)
            
            # Get LLM response
            response = self._call_llm(prompt)
            
            # Parse LLM response
            analysis = self._parse_llm_response(response)
            
            # Add financial news specific fields
            analysis['news_sentiment'] = analysis.get('llm_sentiment', 'Neutral')
            analysis['news_confidence'] = analysis.get('llm_confidence', 50)
            analysis['source_breakdown'] = self._analyze_news_sources(news_headlines)
            analysis['sentiment_indicators'] = self._detect_sentiment_patterns(news_headlines)
            
            return analysis
            
        except Exception as e:
            print(f"❌ Financial news LLM analysis error: {e}")
            return {"analysis": f"Financial news analysis error: {str(e)}", "confidence": 0}
    
    def _create_market_analysis_prompt(self, price_data: Dict, technical_analysis: Dict) -> str:
        """Create a simplified prompt for more reliable parsing"""
        
        current_price = price_data.get('price', 0)
        trend = technical_analysis.get('trend', 'unknown')
        rsi = technical_analysis.get('indicators', {}).get('rsi', 50)
        signals = technical_analysis.get('signals', [])
        
        prompt = f"""Analyze Bitcoin market conditions:

Price: ${current_price:,.2f}
Trend: {trend}
RSI: {rsi:.1f}
Signals: {', '.join(signals) if signals else 'None'}

Provide analysis as simple JSON:
{{
"sentiment": "Bullish",
"recommendation": "HOLD",
"confidence": 75,
"risk_level": "High",
"reasoning": "Brief analysis in one sentence"
}}

Rules:
- sentiment: Bullish, Bearish, or Neutral
- recommendation: BUY, SELL, or HOLD
- confidence: 0-100 number
- risk_level: Low, Medium, or High
- reasoning: Keep under 100 characters

Consider: RSI >70 = overbought, RSI <30 = oversold, Price >$100k = high risk."""

        return prompt
    
    def _create_financial_news_analysis_prompt(self, headlines: List[str], technical_analysis: Dict) -> str:
        """Create prompt that includes financial news sentiment"""
        
        current_price = technical_analysis.get('price', 0)
        trend = technical_analysis.get('trend', 'unknown')
        rsi = technical_analysis.get('indicators', {}).get('rsi', 50)
        
        # Group headlines by source
        polygon_headlines = [h for h in headlines if '[Polygon.io]' in h]
        coindesk_headlines = [h for h in headlines if '[CoinDesk]' in h]
        alpaca_headlines = [h for h in headlines if '[Alpaca]' in h]
        
        prompt = f"""Analyze Bitcoin market using technical data + financial news sentiment:

TECHNICAL DATA:
- Price: ${current_price:,.2f}
- Trend: {trend}
- RSI: {rsi:.1f}

FINANCIAL NEWS HEADLINES:

Polygon.io News ({len(polygon_headlines)} items):
{chr(10).join(polygon_headlines[:5])}

CoinDesk News ({len(coindesk_headlines)} items):
{chr(10).join(coindesk_headlines[:5])}

Alpaca News ({len(alpaca_headlines)} items):
{chr(10).join(alpaca_headlines[:5])}

Provide analysis as JSON:
{{
"sentiment": "Bullish/Bearish/Neutral",
"recommendation": "BUY/SELL/HOLD",
"confidence": 75,
"risk_level": "Low/Medium/High",
"news_sentiment": "Bullish/Bearish/Neutral",
"news_confidence": 80,
"source_analysis": {{
  "polygon": "sentiment",
  "coindesk": "sentiment", 
  "alpaca": "sentiment"
}},
"key_themes": ["institutional adoption", "technical analysis"],
"reasoning": "Combined technical and financial news analysis"
}}

Focus on:
- News sentiment vs price action alignment
- Institutional vs retail sentiment indicators
- Technical confirmation of news-driven moves
- Risk factors from regulatory or market news"""

        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Make API call to OpenRouter"""
        if not self.api_key:
            raise Exception("No OpenRouter API key available")
        
        # Clean the API key - remove any whitespace
        clean_api_key = self.api_key.strip()
        
        headers = {
            "Authorization": f"Bearer {clean_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://trading-bot.local",
            "X-Title": "Crypto Trading Bot"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 800,
            "top_p": 0.9
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                return self._clean_response(content)
            elif response.status_code == 401:
                # Try alternative authentication method
                alt_headers = {
                    "Authorization": f"Bearer {clean_api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "CryptoTradingBot/1.0"
                }
                
                alt_response = requests.post(
                    self.base_url,
                    headers=alt_headers,
                    json=data,
                    timeout=30
                )
                
                if alt_response.status_code == 200:
                    result = alt_response.json()
                    content = result['choices'][0]['message']['content']
                    return self._clean_response(content)
                else:
                    error_text = alt_response.text
                    raise Exception(f"OpenRouter API error: {alt_response.status_code} - {error_text}")
            else:
                error_text = response.text
                raise Exception(f"OpenRouter API error: {response.status_code} - {error_text}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OpenRouter API request failed: {str(e)}")
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response and extract structured data"""
        try:
            # First try simple JSON extraction
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                
                # Try to parse JSON
                try:
                    analysis = json.loads(json_str)
                    
                    # Validate and return if successful
                    if isinstance(analysis, dict):
                        return {
                            'llm_sentiment': analysis.get('sentiment', 'Neutral'),
                            'llm_recommendation': analysis.get('recommendation', 'HOLD'),
                            'llm_confidence': analysis.get('confidence', 50),
                            'llm_risk_level': analysis.get('risk_level', 'Medium'),
                            'llm_reasoning': analysis.get('reasoning', ''),
                            'llm_news_sentiment': analysis.get('news_sentiment', 'Neutral'),
                            'llm_news_confidence': analysis.get('news_confidence', 50),
                            'llm_source_analysis': analysis.get('source_analysis', {}),
                            'llm_key_themes': analysis.get('key_themes', []),
                            'llm_raw_response': response[:300] + '...' if len(response) > 300 else response
                        }
                except json.JSONDecodeError:
                    pass  # Will fall through to text parsing
            
            # Fallback: extract key information using simple text parsing
            return self._simple_text_parse(response)
            
        except Exception as e:
            print(f"⚠️ Error parsing LLM response: {e}")
            return {
                'llm_sentiment': 'Neutral',
                'llm_recommendation': 'HOLD',
                'llm_confidence': 30,
                'llm_reasoning': 'LLM response parsing failed',
                'llm_raw_response': response[:200] + '...' if len(response) > 200 else response
            }
    
    def _simple_text_parse(self, response: str) -> Dict:
        """Simple text parsing when JSON fails"""
        response_lower = response.lower()
        
        # Extract recommendation
        if 'buy' in response_lower and 'sell' not in response_lower:
            recommendation = 'BUY'
        elif 'sell' in response_lower and 'buy' not in response_lower:
            recommendation = 'SELL'
        else:
            recommendation = 'HOLD'
        
        # Extract sentiment
        if 'bullish' in response_lower:
            sentiment = 'Bullish'
        elif 'bearish' in response_lower:
            sentiment = 'Bearish'
        else:
            sentiment = 'Neutral'
        
        # Extract confidence (look for numbers)
        import re
        confidence_match = re.search(r'(\d+)%?', response)
        confidence = int(confidence_match.group(1)) if confidence_match else 50
        confidence = max(0, min(100, confidence))  # Clamp between 0-100
        
        # Extract risk level
        risk_level = 'Medium'
        if 'high risk' in response_lower or 'extreme' in response_lower:
            risk_level = 'High'
        elif 'low risk' in response_lower:
            risk_level = 'Low'
        
        return {
            'llm_sentiment': sentiment,
            'llm_recommendation': recommendation,
            'llm_confidence': confidence,
            'llm_risk_level': risk_level,
            'llm_reasoning': response[:200] + '...' if len(response) > 200 else response,
            'llm_raw_response': response
        }
    
    def _clean_response(self, response: str) -> str:
        """Clean response text to fix parsing issues"""
        import re
        
        # Remove control characters (ASCII 0-31 except newline, tab, carriage return)
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', response)
        
        # Replace problematic characters
        cleaned = cleaned.replace('\n', ' ')  # Replace newlines with spaces
        cleaned = cleaned.replace('\t', ' ')  # Replace tabs with spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Collapse multiple spaces
        
        # Remove any remaining problematic characters for JSON
        cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in ['\n', '\t'])
        
        return cleaned.strip()
    
    def _analyze_news_sources(self, headlines: List[str]) -> Dict:
        """Analyze sentiment by news source"""
        source_sentiment = {
            'polygon': {'positive': 0, 'negative': 0, 'neutral': 0},
            'coindesk': {'positive': 0, 'negative': 0, 'neutral': 0},
            'alpaca': {'positive': 0, 'negative': 0, 'neutral': 0}
        }
        
        positive_keywords = ['bull', 'bullish', 'rise', 'up', 'gain', 'profit', 'positive', 'adoption', 'breakthrough']
        negative_keywords = ['bear', 'bearish', 'fall', 'down', 'loss', 'negative', 'crash', 'decline', 'concern']
        
        for headline in headlines:
            headline_lower = headline.lower()
            
            # Determine source
            source = None
            if '[polygon.io]' in headline_lower:
                source = 'polygon'
            elif '[coindesk]' in headline_lower:
                source = 'coindesk'
            elif '[alpaca]' in headline_lower:
                source = 'alpaca'
            
            if source:
                # Analyze sentiment
                positive_score = sum(1 for word in positive_keywords if word in headline_lower)
                negative_score = sum(1 for word in negative_keywords if word in headline_lower)
                
                if positive_score > negative_score:
                    source_sentiment[source]['positive'] += 1
                elif negative_score > positive_score:
                    source_sentiment[source]['negative'] += 1
                else:
                    source_sentiment[source]['neutral'] += 1
        
        return source_sentiment
    
    def _detect_sentiment_patterns(self, headlines: List[str]) -> List[str]:
        """Detect sentiment patterns in financial news"""
        sentiment_indicators = []
        
        # Count sentiment emojis
        bullish_count = sum(1 for h in headlines if '📈' in h)
        bearish_count = sum(1 for h in headlines if '📉' in h)
        neutral_count = sum(1 for h in headlines if '➡️' in h)
        
        total_headlines = len(headlines)
        if total_headlines > 0:
            if bullish_count / total_headlines > 0.6:
                sentiment_indicators.append("Strong bullish news sentiment")
            elif bearish_count / total_headlines > 0.6:
                sentiment_indicators.append("Strong bearish news sentiment")
            elif neutral_count / total_headlines > 0.7:
                sentiment_indicators.append("Neutral market news")
        
        # Check for institutional keywords
        institutional_keywords = ['institutional', 'corporate', 'etf', 'regulation', 'sec', 'federal']
        institutional_mentions = sum(1 for h in headlines for keyword in institutional_keywords if keyword in h.lower())
        
        if institutional_mentions > 2:
            sentiment_indicators.append("High institutional focus")
        
        # Check for technical analysis mentions
        technical_keywords = ['technical', 'resistance', 'support', 'pattern', 'chart']
        technical_mentions = sum(1 for h in headlines for keyword in technical_keywords if keyword in h.lower())
        
        if technical_mentions > 1:
            sentiment_indicators.append("Technical analysis focus")
        
        return sentiment_indicators[:5]  # Top 5 indicators
    
    def analyze_news_sentiment(self, news_headlines: List[str]) -> Dict:
        """Analyze news sentiment impact on Bitcoin"""
        if not self.api_key or not news_headlines:
            return {"news_sentiment": "Neutral", "impact": "Low"}
        
        try:
            headlines_text = '\n'.join(f"- {headline}" for headline in news_headlines[:10])
            
            prompt = f"""Analyze the sentiment of these recent Bitcoin/cryptocurrency financial news headlines and their potential market impact:

{headlines_text}

Provide analysis in JSON format:
{{
  "overall_sentiment": "Bullish/Bearish/Neutral",
  "impact_level": "Low/Medium/High",
  "key_themes": ["theme1", "theme2"],
  "market_impact": "Brief explanation of likely price impact",
  "confidence": 75
}}"""

            response = self._call_llm(prompt)
            analysis = self._parse_llm_response(response)
            
            return {
                'news_sentiment': analysis.get('llm_sentiment', 'Neutral'),
                'news_impact': analysis.get('llm_confidence', 50),
                'news_reasoning': analysis.get('llm_reasoning', '')
            }
            
        except Exception as e:
            print(f"❌ News sentiment analysis error: {e}")
            return {"news_sentiment": "Neutral", "impact": "Low"}
    
    def get_risk_assessment(self, portfolio_data: Dict, market_data: Dict) -> Dict:
        """Get LLM-based risk assessment"""
        if not self.api_key:
            return {"risk_level": "Medium", "suggestions": []}
        
        try:
            prompt = f"""As a risk management expert, assess the current trading risk based on:

PORTFOLIO:
- Current Position: {portfolio_data.get('position_type', 'None')}
- Position Size: {portfolio_data.get('position_size', 0)}
- Unrealized PnL: {portfolio_data.get('unrealized_pnl', 0)}
- Available Balance: {portfolio_data.get('available_balance', 0)}

MARKET CONDITIONS:
- Price: ${market_data.get('price', 0):,.2f}
- Volatility: {market_data.get('volatility', 'Unknown')}
- Trend: {market_data.get('trend', 'Unknown')}

Provide risk assessment in JSON format:
{{
  "risk_level": "Low/Medium/High/Extreme",
  "risk_score": 65,
  "risk_factors": ["factor1", "factor2"],
  "recommendations": ["action1", "action2"],
  "position_sizing": "Suggested position size guidance",
  "stop_loss": "Stop loss recommendation"
}}"""

            response = self._call_llm(prompt)
            return self._parse_llm_response(response)
            
        except Exception as e:
            print(f"❌ Risk assessment error: {e}")
            return {"risk_level": "Medium", "suggestions": []}

# Enhanced NewsFetcher integration - backward compatibility
class NewsFetcher:
    """News fetcher class for backward compatibility"""
    
    def __init__(self):
        if NEWS_FETCHER_AVAILABLE:
            self.enhanced_fetcher = EnhancedNewsFetcher()
        else:
            self.enhanced_fetcher = None
            print("⚠️ Enhanced news fetcher not available")
    
    def get_bitcoin_news(self, limit: int = 15) -> List[str]:
        """Get Bitcoin news from financial sources"""
        if self.enhanced_fetcher:
            try:
                return self.enhanced_fetcher.get_bitcoin_news(limit)
            except Exception as e:
                print(f"❌ Enhanced news fetch error: {e}")
                return self._get_fallback_headlines()
        else:
            return self._get_fallback_headlines()
    
    def _get_fallback_headlines(self) -> List[str]:
        """Fallback headlines when enhanced fetcher is unavailable"""
        return [
            "[Financial] Bitcoin consolidates above key support levels 📈",
            "[Financial] Cryptocurrency market shows institutional interest ➡️",
            "[Financial] Technical indicators suggest potential breakout 📈",
            "[Financial] Regulatory clarity discussions continue in markets ➡️",
            "[Financial] Bitcoin network fundamentals remain strong 📈"
        ]

# Usage example and testing
if __name__ == "__main__":
    # Test the enhanced LLM integration
    analyzer = LLMAnalyzer()
    news_fetcher = NewsFetcher()
    
    # Get financial news headlines
    headlines = news_fetcher.get_bitcoin_news()
    print("📰 Financial News Headlines:")
    for i, headline in enumerate(headlines, 1):
        print(f"{i}. {headline}")
    
    # Sample technical analysis
    technical_analysis = {
        'price': 103963.10,
        'trend': 'bullish',
        'signals': ['RSI_OVERBOUGHT'],
        'indicators': {'rsi': 95.38, 'macd': -14.0962}
    }
    
    # Get enhanced LLM analysis with financial news
    if headlines and analyzer.api_key:
        result = analyzer.analyze_financial_news_sentiment(headlines, technical_analysis)
        print("\n🤖 Enhanced LLM Analysis (Technical + Financial News):")
        print(json.dumps(result, indent=2))
    else:
        # Fallback to regular analysis
        price_data = {'price': 103963.10}
        result = analyzer.analyze_market_sentiment(price_data, technical_analysis)
        print("\n🤖 Standard LLM Analysis:")
        print(json.dumps(result, indent=2))
