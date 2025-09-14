#!/usr/bin/env python3
"""
Local FinBERT Implementation for AlphaPlus
Downloads and tests FinBERT model locally for financial sentiment analysis
"""

import asyncio
import sys
import os
import logging
from datetime import datetime
from typing import Dict, List, Any

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LocalFinBERTManager:
    """Local FinBERT implementation for financial sentiment analysis"""
    
    def __init__(self):
        self.model = None
        self.pipeline = None
        self.model_loaded = False
        self.model_name = "ProsusAI/finbert"
        
    def load_model(self):
        """Load FinBERT model locally"""
        try:
            print("🤗 Loading FinBERT model locally...")
            print(f"   Model: {self.model_name}")
            print("   This may take a few minutes on first run...")
            
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            
            # Load FinBERT pipeline
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=-1,  # CPU only for compatibility
                return_all_scores=True
            )
            
            self.model_loaded = True
            print("✅ FinBERT model loaded successfully!")
            print("   Ready for financial sentiment analysis")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading FinBERT model: {e}")
            print("   Falling back to keyword analysis")
            self.model_loaded = False
            return False
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze financial sentiment using FinBERT"""
        if not self.model_loaded:
            return self._fallback_sentiment_analysis(text)
        
        try:
            # Clean and prepare text
            clean_text = self._clean_text(text)
            
            # Get sentiment prediction
            results = self.pipeline(clean_text)
            
            # Process results
            sentiment_scores = {}
            for result in results[0]:
                label = result['label'].lower()
                score = result['score']
                sentiment_scores[label] = score
            
            # Determine overall sentiment
            overall_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            confidence = sentiment_scores[overall_sentiment]
            
            # Convert to our format
            sentiment_mapping = {
                'positive': 'bullish',
                'negative': 'bearish',
                'neutral': 'neutral'
            }
            
            return {
                'sentiment': sentiment_mapping.get(overall_sentiment, 'neutral'),
                'confidence': confidence,
                'scores': sentiment_scores,
                'model': 'finbert_local',
                'text_analyzed': clean_text[:100] + "..." if len(clean_text) > 100 else clean_text
            }
            
        except Exception as e:
            print(f"⚠️ FinBERT analysis failed: {e}")
            return self._fallback_sentiment_analysis(text)
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple texts for sentiment"""
        results = []
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)
        return results
    
    def _clean_text(self, text: str) -> str:
        """Clean text for FinBERT analysis"""
        import re
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Limit length for FinBERT (max 512 tokens)
        if len(text) > 500:
            text = text[:500] + "..."
        
        return text
    
    def _fallback_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Fallback keyword-based sentiment analysis"""
        text_lower = text.lower()
        
        # Financial sentiment keywords
        bullish_keywords = [
            'bullish', 'moon', 'pump', 'surge', 'rally', 'adoption', 'partnership',
            'breakthrough', 'innovation', 'growth', 'profit', 'gain', 'rise', 'up',
            'buy', 'long', 'hodl', 'diamond hands', 'to the moon'
        ]
        
        bearish_keywords = [
            'bearish', 'dump', 'crash', 'sell', 'fud', 'scam', 'regulation',
            'decline', 'loss', 'drop', 'down', 'short', 'panic', 'fear',
            'correction', 'bubble', 'overvalued', 'sell off'
        ]
        
        bullish_count = sum(1 for keyword in bullish_keywords if keyword in text_lower)
        bearish_count = sum(1 for keyword in bearish_keywords if keyword in text_lower)
        
        if bullish_count > bearish_count:
            sentiment = 'bullish'
            confidence = min(0.7, 0.5 + (bullish_count * 0.1))
        elif bearish_count > bullish_count:
            sentiment = 'bearish'
            confidence = min(0.7, 0.5 + (bearish_count * 0.1))
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'scores': {'bullish': bullish_count, 'bearish': bearish_count, 'neutral': 1},
            'model': 'keyword_fallback',
            'text_analyzed': text[:100] + "..." if len(text) > 100 else text
        }

async def test_finbert_local():
    """Test local FinBERT implementation"""
    print("🚀 Testing Local FinBERT Implementation")
    print("=" * 60)
    
    # Initialize FinBERT manager
    finbert_manager = LocalFinBERTManager()
    
    # Load model
    print("\n📥 Loading FinBERT model...")
    model_loaded = finbert_manager.load_model()
    
    if not model_loaded:
        print("⚠️ Model loading failed, using fallback analysis")
    
    # Test cases
    test_cases = [
        "Bitcoin is going to the moon! 🚀 This is just the beginning of a massive bull run.",
        "The crypto market is crashing hard. Time to sell everything before it's too late.",
        "Bitcoin price remains stable around $50,000 with moderate trading volume.",
        "Ethereum's new upgrade will revolutionize DeFi and drive massive adoption.",
        "Regulatory concerns are causing panic selling across all cryptocurrencies.",
        "The market is showing mixed signals with both positive and negative indicators."
    ]
    
    print(f"\n🧪 Testing {len(test_cases)} financial text samples...")
    print("-" * 60)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}:")
        print(f"   Text: {text[:80]}...")
        
        result = finbert_manager.analyze_sentiment(text)
        
        print(f"   ✅ Sentiment: {result['sentiment']}")
        print(f"   📊 Confidence: {result['confidence']:.3f}")
        print(f"   🤖 Model: {result['model']}")
        
        if 'scores' in result:
            print(f"   📈 Scores: {result['scores']}")
    
    # Test batch analysis
    print(f"\n🔄 Testing batch analysis...")
    batch_results = finbert_manager.analyze_batch(test_cases[:3])
    
    print(f"   ✅ Batch analysis completed: {len(batch_results)} results")
    for i, result in enumerate(batch_results):
        print(f"   {i+1}. {result['sentiment']} (confidence: {result['confidence']:.3f})")
    
    print(f"\n🎉 Local FinBERT testing completed!")
    print(f"   Model loaded: {model_loaded}")
    print(f"   Fallback available: {not model_loaded}")
    
    return finbert_manager

async def main():
    """Main test function"""
    try:
        finbert_manager = await test_finbert_local()
        
        print(f"\n💡 USAGE EXAMPLES:")
        print("-" * 40)
        print("# Initialize FinBERT")
        print("finbert = LocalFinBERTManager()")
        print("finbert.load_model()")
        print("")
        print("# Analyze sentiment")
        print("result = finbert.analyze_sentiment('Bitcoin is bullish!')")
        print("print(result['sentiment'])  # 'bullish'")
        print("print(result['confidence'])  # 0.85")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        logger.error(f"FinBERT test error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
