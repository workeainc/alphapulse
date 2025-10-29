"""
LLM Threshold Predictor for AlphaPulse
Lightweight 3B-parameter LLM for threshold decisions using Hugging Face transformers
"""

import asyncio
import logging
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import hashlib
from collections import OrderedDict

# Hugging Face imports
try:
    from transformers import (
        AutoModelForSequenceClassification, 
        AutoTokenizer, 
        AutoModelForCausalLM,
        pipeline,
        BitsAndBytesConfig
    )
    from torch.quantization import quantize_dynamic
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available, LLM threshold predictor disabled")

logger = logging.getLogger(__name__)

@dataclass
class MarketContext:
    """Market context for LLM input"""
    market_state: str  # bullish, bearish, neutral
    volume: float
    volatility: float
    trend_strength: float
    recent_performance: List[float]
    current_threshold: float
    market_regime: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ThresholdPrediction:
    """LLM threshold prediction result"""
    volume_threshold: float
    trend_threshold: float
    confidence_threshold: float
    prediction_confidence: float
    reasoning: str
    processing_time: float
    model_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class LLMThresholdPredictor:
    """
    Lightweight 3B-parameter LLM for threshold decisions.
    Uses Hugging Face transformers with quantization for low latency.
    """
    
    def __init__(self, 
                 model_name: str = "distilbert-base-uncased",
                 use_quantization: bool = True,
                 cache_size: int = 1000,
                 max_input_length: int = 512):
        
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.max_input_length = max_input_length
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        # Cache for LLM outputs
        self.cache = OrderedDict()
        self.cache_size = cache_size
        
        # Performance tracking
        self.total_predictions = 0
        self.cache_hits = 0
        self.avg_processing_time = 0.0
        
        # Model loading
        self._load_model()
        
        logger.info(f"LLM Threshold Predictor initialized with model: {model_name}")
    
    def _load_model(self):
        """Load the Hugging Face model and tokenizer"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, using fallback predictor")
            return
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            if self.model_name == "distilbert-base-uncased":
                # Use sequence classification for threshold prediction
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=3  # volume, trend, confidence thresholds
                )
            else:
                # Use causal LM for text generation
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Apply quantization if requested
            if self.use_quantization and torch.cuda.is_available():
                self._apply_quantization()
            
            # Create pipeline for easier inference
            if self.model_name == "distilbert-base-uncased":
                self.pipeline = pipeline(
                    "text-classification",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if torch.cuda.is_available() else -1
                )
            
            logger.info(f"Model {self.model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            self.model = None
            self.tokenizer = None
            self.pipeline = None
    
    def _apply_quantization(self):
        """Apply quantization to reduce model size and improve inference speed"""
        try:
            if self.model is not None:
                # Dynamic quantization for CPU inference
                self.model = quantize_dynamic(
                    self.model, 
                    {torch.nn.Linear}, 
                    dtype=torch.qint8
                )
                logger.info("Model quantized successfully")
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
    
    async def predict_thresholds(self, market_context: MarketContext) -> ThresholdPrediction:
        """
        Predict optimal thresholds using LLM
        
        Args:
            market_context: Current market context
            
        Returns:
            ThresholdPrediction with predicted thresholds
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(market_context)
        if cache_key in self.cache:
            self.cache_hits += 1
            cached_result = self.cache[cache_key]
            cached_result.processing_time = time.time() - start_time
            return cached_result
        
        try:
            if self.model is None or self.tokenizer is None:
                # Fallback to heuristic prediction
                return await self._heuristic_prediction(market_context)
            
            # Prepare input text
            input_text = self._prepare_input_text(market_context)
            
            # Get prediction
            if self.model_name == "distilbert-base-uncased":
                prediction = await self._predict_with_classification(input_text)
            else:
                prediction = await self._predict_with_generation(input_text)
            
            # Create result
            result = ThresholdPrediction(
                volume_threshold=prediction['volume_threshold'],
                trend_threshold=prediction['trend_threshold'],
                confidence_threshold=prediction['confidence_threshold'],
                prediction_confidence=prediction['confidence'],
                reasoning=prediction.get('reasoning', ''),
                processing_time=time.time() - start_time,
                model_used=self.model_name,
                metadata=prediction.get('metadata', {})
            )
            
            # Cache result
            self._cache_result(cache_key, result)
            
            # Update statistics
            self.total_predictions += 1
            self.avg_processing_time = (
                (self.avg_processing_time * (self.total_predictions - 1) + result.processing_time) 
                / self.total_predictions
            )
            
            return result
            
        except Exception as e:
            logger.error(f"LLM prediction error: {e}")
            return await self._heuristic_prediction(market_context)
    
    async def _predict_with_classification(self, input_text: str) -> Dict[str, Any]:
        """Predict thresholds using sequence classification model"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_input_length
            )
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Convert logits to probabilities
                probabilities = torch.softmax(logits, dim=1)
                
                # Map to threshold values
                volume_threshold = self._map_probability_to_volume(probabilities[0][0].item())
                trend_threshold = self._map_probability_to_trend(probabilities[0][1].item())
                confidence_threshold = self._map_probability_to_confidence(probabilities[0][2].item())
                
                # Calculate overall confidence
                confidence = torch.max(probabilities).item()
                
                return {
                    'volume_threshold': volume_threshold,
                    'trend_threshold': trend_threshold,
                    'confidence_threshold': confidence_threshold,
                    'confidence': confidence,
                    'reasoning': f"Classification-based prediction with confidence {confidence:.3f}",
                    'metadata': {
                        'probabilities': probabilities.tolist(),
                        'model_type': 'classification'
                    }
                }
                
        except Exception as e:
            logger.error(f"Classification prediction error: {e}")
            raise
    
    async def _predict_with_generation(self, input_text: str) -> Dict[str, Any]:
        """Predict thresholds using text generation model"""
        try:
            # Prepare prompt
            prompt = self._create_generation_prompt(input_text)
            
            # Generate response
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_input_length
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode response
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Parse response
                thresholds = self._parse_generated_response(response)
                
                return {
                    'volume_threshold': thresholds['volume'],
                    'trend_threshold': thresholds['trend'],
                    'confidence_threshold': thresholds['confidence'],
                    'confidence': thresholds.get('confidence', 0.7),
                    'reasoning': response,
                    'metadata': {
                        'generated_text': response,
                        'model_type': 'generation'
                    }
                }
                
        except Exception as e:
            logger.error(f"Generation prediction error: {e}")
            raise
    
    def _prepare_input_text(self, market_context: MarketContext) -> str:
        """Prepare input text for the LLM"""
        # Format market context as structured text
        recent_perf_mean = np.mean(market_context.recent_performance) if market_context.recent_performance else 0.5
        recent_perf_std = np.std(market_context.recent_performance) if market_context.recent_performance else 0.1
        
        input_text = (
            f"Market: {market_context.market_state}, "
            f"Volume: {market_context.volume:.2f}, "
            f"Volatility: {market_context.volatility:.4f}, "
            f"Trend Strength: {market_context.trend_strength:.3f}, "
            f"Regime: {market_context.market_regime}, "
            f"Recent Performance: {recent_perf_mean:.3f} ± {recent_perf_std:.3f}, "
            f"Current Threshold: {market_context.current_threshold:.3f}"
        )
        
        return input_text
    
    def _create_generation_prompt(self, input_text: str) -> str:
        """Create prompt for text generation model"""
        prompt = (
            f"Given the following market conditions:\n{input_text}\n\n"
            f"Please suggest optimal trading thresholds in the format:\n"
            f"Volume Threshold: [value]\n"
            f"Trend Threshold: [value]\n"
            f"Confidence Threshold: [value]\n"
            f"Reasoning: [explanation]\n\n"
            f"Response:"
        )
        return prompt
    
    def _parse_generated_response(self, response: str) -> Dict[str, float]:
        """Parse generated response to extract thresholds"""
        try:
            # Extract values using simple parsing
            lines = response.split('\n')
            thresholds = {}
            
            for line in lines:
                line = line.strip().lower()
                if 'volume threshold:' in line:
                    value = self._extract_number(line)
                    if value is not None:
                        thresholds['volume'] = value
                elif 'trend threshold:' in line:
                    value = self._extract_number(line)
                    if value is not None:
                        thresholds['trend'] = value
                elif 'confidence threshold:' in line:
                    value = self._extract_number(line)
                    if value is not None:
                        thresholds['confidence'] = value
            
            # Set defaults if not found
            thresholds.setdefault('volume', 500.0)
            thresholds.setdefault('trend', 0.5)
            thresholds.setdefault('confidence', 0.6)
            
            return thresholds
            
        except Exception as e:
            logger.error(f"Response parsing error: {e}")
            return {'volume': 500.0, 'trend': 0.5, 'confidence': 0.6}
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract number from text"""
        import re
        numbers = re.findall(r'[-+]?\d*\.?\d+', text)
        return float(numbers[0]) if numbers else None
    
    def _map_probability_to_volume(self, prob: float) -> float:
        """Map probability to volume threshold"""
        return prob * 1000.0  # Map [0,1] to [0,1000]
    
    def _map_probability_to_trend(self, prob: float) -> float:
        """Map probability to trend threshold"""
        return prob  # Map [0,1] to [0,1]
    
    def _map_probability_to_confidence(self, prob: float) -> float:
        """Map probability to confidence threshold"""
        return 0.1 + prob * 0.8  # Map [0,1] to [0.1,0.9]
    
    async def _heuristic_prediction(self, market_context: MarketContext) -> ThresholdPrediction:
        """Fallback heuristic prediction when LLM is not available"""
        # Simple heuristic based on market context
        base_volume = 500.0
        base_trend = 0.5
        base_confidence = 0.6
        
        # Adjust based on market state
        if market_context.market_state == "bullish":
            base_volume += 100
            base_trend += 0.1
            base_confidence += 0.05
        elif market_context.market_state == "bearish":
            base_volume -= 100
            base_trend -= 0.1
            base_confidence += 0.1  # Higher confidence for bearish markets
        
        # Adjust based on volatility
        if market_context.volatility > 0.03:
            base_confidence += 0.1  # Higher threshold for high volatility
        
        # Adjust based on recent performance
        if market_context.recent_performance:
            avg_performance = np.mean(market_context.recent_performance)
            if avg_performance < 0.4:
                base_confidence -= 0.05  # Lower threshold if performance is poor
            elif avg_performance > 0.8:
                base_confidence += 0.05  # Higher threshold if performance is excellent
        
        return ThresholdPrediction(
            volume_threshold=max(100.0, min(1000.0, base_volume)),
            trend_threshold=max(0.1, min(0.9, base_trend)),
            confidence_threshold=max(0.1, min(0.9, base_confidence)),
            prediction_confidence=0.5,
            reasoning="Heuristic fallback prediction",
            processing_time=0.001,
            model_used="heuristic",
            metadata={'method': 'heuristic_fallback'}
        )
    
    def _generate_cache_key(self, market_context: MarketContext) -> str:
        """Generate cache key for market context"""
        # Create hash of market context
        context_str = (
            f"{market_context.market_state}_{market_context.volume:.2f}_"
            f"{market_context.volatility:.4f}_{market_context.trend_strength:.3f}_"
            f"{market_context.market_regime}_{market_context.current_threshold:.3f}"
        )
        return hashlib.md5(context_str.encode()).hexdigest()
    
    def _cache_result(self, key: str, result: ThresholdPrediction):
        """Cache prediction result"""
        # Remove oldest entry if cache is full
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
        
        self.cache[key] = result
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'total_predictions': self.total_predictions,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / max(1, self.total_predictions),
            'avg_processing_time': self.avg_processing_time,
            'cache_size': len(self.cache),
            'model_loaded': self.model is not None,
            'quantization_enabled': self.use_quantization
        }
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()
        logger.info("LLM predictor cache cleared")

# Global LLM threshold predictor instance
llm_threshold_predictor = LLMThresholdPredictor(
    model_name="distilbert-base-uncased",
    use_quantization=True,
    cache_size=1000
)
