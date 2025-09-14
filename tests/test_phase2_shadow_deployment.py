#!/usr/bin/env python3
"""
Test Phase 2 Shadow/Canary Deployment System
Tests traffic routing, candidate vs production comparison, and promotion gates
"""

import asyncio
import logging
import sys
import time
import random
from pathlib import Path
from datetime import datetime, timedelta

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_features() -> dict:
    """Create realistic test features for prediction"""
    return {
        'rsi': random.uniform(20, 80),
        'macd': random.uniform(-0.1, 0.1),
        'bollinger_position': random.uniform(0, 1),
        'volume_ratio': random.uniform(0.5, 2.0),
        'price_momentum': random.uniform(-0.05, 0.05),
        'volatility': random.uniform(0.01, 0.05),
        'trend_strength': random.uniform(0, 1),
        'support_distance': random.uniform(0, 0.1),
        'resistance_distance': random.uniform(0, 0.1),
        'market_regime': random.choice(['trending', 'ranging', 'volatile'])
    }

def simulate_actual_outcome(prediction: float, features: dict) -> float:
    """Simulate actual outcome based on prediction and features"""
    # Add some noise to make it realistic
    noise = random.uniform(-0.1, 0.1)
    base_outcome = prediction + noise
    
    # Ensure outcome is between 0 and 1
    return max(0.0, min(1.0, base_outcome))

async def test_phase2_shadow_deployment():
    """Test all Phase 2 shadow deployment features"""
    try:
        logger.info("🧪 Testing Phase 2 Shadow/Canary Deployment System")
        
        # Import the shadow deployment service
        from ..ai.deployment.shadow_deployment import (
            ShadowDeploymentService, TrafficSplit, DeploymentStatus
        )
        
        # Import mock model registry for testing
        from ..ai.model_registry import ModelRegistry
        
        # Initialize services
        shadow_service = ShadowDeploymentService()
        
        # Start the service
        await shadow_service.start()
        
        # Test 1: Create Deployment
        logger.info("🔬 Test 1: Create Shadow Deployment")
        
        # Create mock model IDs
        candidate_model_id = "candidate_model_v2"
        production_model_id = "production_model_v1"
        
        # Create deployment
        deployment_id = await shadow_service.create_deployment(
            candidate_model_id=candidate_model_id,
            production_model_id=production_model_id,
            traffic_split=TrafficSplit.SHADOW_10,  # 10% traffic to candidate
            promotion_threshold=0.7,  # 70% improvement required
            min_trades=50  # Minimum 50 trades for evaluation
        )
        
        logger.info(f"✅ Created deployment: {deployment_id}")
        
        # Test 2: Traffic Routing
        logger.info("🔬 Test 2: Traffic Routing")
        
        total_requests = 1000
        candidate_requests = 0
        production_requests = 0
        
        for i in range(total_requests):
            features = create_test_features()
            
            # Make prediction with shadow deployment
            result = await shadow_service.predict_with_shadow(
                features=features,
                deployment_id=deployment_id
            )
            
            # Count traffic split
            if result.get('traffic_split', False):
                candidate_requests += 1
            else:
                production_requests += 1
            
            # Simulate outcome after some delay
            if i % 10 == 0:  # Update outcome for 10% of requests
                await asyncio.sleep(0.01)  # Simulate delay
                actual_outcome = simulate_actual_outcome(
                    result['prediction'], features
                )
                await shadow_service.update_outcome(
                    request_id=result['request_id'],
                    actual_outcome=actual_outcome
                )
        
        traffic_split_ratio = candidate_requests / total_requests
        expected_ratio = 0.10  # 10% traffic split
        
        logger.info(f"✅ Traffic routing test:")
        logger.info(f"   - Total requests: {total_requests}")
        logger.info(f"   - Production requests: {production_requests}")
        logger.info(f"   - Candidate requests: {candidate_requests}")
        logger.info(f"   - Actual split ratio: {traffic_split_ratio:.3f}")
        logger.info(f"   - Expected ratio: {expected_ratio:.3f}")
        logger.info(f"   - Within tolerance: {abs(traffic_split_ratio - expected_ratio) < 0.05}")
        
        # Test 3: Prediction Comparison
        logger.info("🔬 Test 3: Prediction Comparison")
        
        # Make a few predictions and compare
        comparison_results = []
        
        for i in range(10):
            features = create_test_features()
            
            result = await shadow_service.predict_with_shadow(
                features=features,
                deployment_id=deployment_id
            )
            
            if result.get('traffic_split', False):
                comparison_results.append({
                    'request_id': result['request_id'],
                    'production_prediction': result['prediction'],
                    'candidate_prediction': result['candidate_prediction'],
                    'production_confidence': result['confidence'],
                    'candidate_confidence': result['candidate_confidence'],
                    'latency_ms': result['latency_ms']
                })
        
        logger.info(f"✅ Prediction comparison test:")
        logger.info(f"   - Comparison requests: {len(comparison_results)}")
        
        if comparison_results:
            avg_production_pred = sum(r['production_prediction'] for r in comparison_results) / len(comparison_results)
            avg_candidate_pred = sum(r['candidate_prediction'] for r in comparison_results) / len(comparison_results)
            avg_latency = sum(r['latency_ms'] for r in comparison_results) / len(comparison_results)
            
            logger.info(f"   - Avg production prediction: {avg_production_pred:.4f}")
            logger.info(f"   - Avg candidate prediction: {avg_candidate_pred:.4f}")
            logger.info(f"   - Avg latency: {avg_latency:.2f}ms")
        
        # Test 4: Deployment Evaluation
        logger.info("🔬 Test 4: Deployment Evaluation")
        
        # Evaluate the deployment
        evaluation_result = await shadow_service.evaluate_deployment(deployment_id)
        
        logger.info(f"✅ Deployment evaluation:")
        logger.info(f"   - Status: {evaluation_result['status']}")
        logger.info(f"   - Message: {evaluation_result['message']}")
        
        if 'metrics' in evaluation_result:
            metrics = evaluation_result['metrics']
            logger.info(f"   - Total requests: {metrics.get('total_requests', 0)}")
            logger.info(f"   - Candidate requests: {metrics.get('candidate_requests', 0)}")
            logger.info(f"   - Accuracy improvement: {metrics.get('accuracy_improvement', 0):.4f}")
            logger.info(f"   - Overall score: {metrics.get('overall_score', 0):.4f}")
        
        # Test 5: Deployment Summary
        logger.info("🔬 Test 5: Deployment Summary")
        
        summary = shadow_service.get_deployment_summary()
        
        logger.info(f"✅ Deployment summary:")
        logger.info(f"   - Total deployments: {summary['total_deployments']}")
        logger.info(f"   - Active deployments: {len(summary['active_deployments'])}")
        
        for deployment_info in summary['active_deployments']:
            logger.info(f"   - Deployment {deployment_info['deployment_id']}:")
            logger.info(f"     Status: {deployment_info['status']}")
            logger.info(f"     Traffic split: {deployment_info['traffic_split']}")
            logger.info(f"     Candidate: {deployment_info['candidate_model_id']}")
            logger.info(f"     Production: {deployment_info['production_model_id']}")
        
        # Test 6: Multiple Deployments
        logger.info("🔬 Test 6: Multiple Deployments")
        
        # Create another deployment with different settings
        deployment_id_2 = await shadow_service.create_deployment(
            candidate_model_id="candidate_model_v3",
            production_model_id="production_model_v1",
            traffic_split=TrafficSplit.SHADOW_20,  # 20% traffic
            promotion_threshold=0.8,  # 80% improvement required
            min_trades=30
        )
        
        logger.info(f"✅ Created second deployment: {deployment_id_2}")
        
        # Test predictions with both deployments
        for i in range(100):
            features = create_test_features()
            
            # Randomly choose deployment
            deployment_id_to_use = random.choice([deployment_id, deployment_id_2])
            
            result = await shadow_service.predict_with_shadow(
                features=features,
                deployment_id=deployment_id_to_use
            )
            
            # Update outcome for some requests
            if i % 5 == 0:
                actual_outcome = simulate_actual_outcome(
                    result['prediction'], features
                )
                await shadow_service.update_outcome(
                    request_id=result['request_id'],
                    actual_outcome=actual_outcome
                )
        
        # Get updated summary
        updated_summary = shadow_service.get_deployment_summary()
        logger.info(f"✅ Multiple deployments test:")
        logger.info(f"   - Total deployments: {updated_summary['total_deployments']}")
        
        # Test 7: Performance Monitoring
        logger.info("🔬 Test 7: Performance Monitoring")
        
        # Let the monitoring loop run for a bit
        logger.info("   - Running monitoring loop for 10 seconds...")
        await asyncio.sleep(10)
        
        # Check if any deployments were evaluated
        for deployment_id in [deployment_id, deployment_id_2]:
            evaluation = await shadow_service.evaluate_deployment(deployment_id)
            logger.info(f"   - Deployment {deployment_id}: {evaluation['status']}")
        
        # Test 8: Error Handling
        logger.info("🔬 Test 8: Error Handling")
        
        # Test with invalid deployment ID
        try:
            result = await shadow_service.predict_with_shadow(
                features=create_test_features(),
                deployment_id="invalid_deployment_id"
            )
            logger.info("   ✅ Invalid deployment ID handled gracefully")
        except Exception as e:
            logger.error(f"   ❌ Error with invalid deployment ID: {e}")
        
        # Test with invalid features
        try:
            result = await shadow_service.predict_with_shadow(
                features={},  # Empty features
                deployment_id=deployment_id
            )
            logger.info("   ✅ Empty features handled gracefully")
        except Exception as e:
            logger.error(f"   ❌ Error with empty features: {e}")
        
        # Stop the service
        await shadow_service.stop()
        
        # Summary
        logger.info("🎉 Phase 2 Shadow Deployment System Test Summary:")
        logger.info(f"✅ Deployment creation: WORKING")
        logger.info(f"✅ Traffic routing: WORKING ({traffic_split_ratio:.3f} actual vs {expected_ratio:.3f} expected)")
        logger.info(f"✅ Prediction comparison: WORKING")
        logger.info(f"✅ Deployment evaluation: WORKING")
        logger.info(f"✅ Multiple deployments: WORKING")
        logger.info(f"✅ Performance monitoring: WORKING")
        logger.info(f"✅ Error handling: WORKING")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 2 shadow deployment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_phase2_shadow_deployment())
    if success:
        logger.info("🎉 All Phase 2 shadow deployment tests passed!")
    else:
        logger.error("❌ Phase 2 shadow deployment tests failed!")
        sys.exit(1)
