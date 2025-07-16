import time
import os
import sys
import json
from rp_handler import handler

def test_normal_request():
    """Test normal request with compression"""
    print("\n" + "="*50)
    print("TEST 1: Normal request with compression")
    print("="*50)
    
    test_event = {
        "input": {
            "prompt": "The capital of France is",
            "model_id": "test-model",
            "batch_size": 48,
            "max_n_logits": 10,
            "desired_logit_prob": 0.95,
            "node_threshold": 0.8,
            "edge_threshold": 0.98,
            "slug_identifier": "test-run",
            "max_feature_nodes": 10000,
            "signed_url": "https://httpbin.org/put",
            "user_id": "test-user",
            "compress": True,
            "deadline": int(time.time()) + 300  # 5 minutes from now
        }
    }
    
    result = handler(test_event)
    print(f"\nTest Result: {result}")
    return result

def test_deadline_exceeded():
    """Test request with expired deadline"""
    print("\n" + "="*50)
    print("TEST 2: Request with expired deadline")
    print("="*50)
    
    test_event = {
        "input": {
            "prompt": "The capital of France is",
            "model_id": "test-model",
            "batch_size": 48,
            "max_n_logits": 10,
            "desired_logit_prob": 0.95,
            "node_threshold": 0.8,
            "edge_threshold": 0.98,
            "slug_identifier": "test-run",
            "max_feature_nodes": 10000,
            "signed_url": "https://httpbin.org/put",
            "user_id": "test-user",
            "compress": False,
            "deadline": int(time.time()) - 60  # 1 minute ago
        }
    }
    
    result = handler(test_event)
    print(f"\nTest Result: {result}")
    return result

def test_no_compression():
    """Test request without compression"""
    print("\n" + "="*50)
    print("TEST 3: Request without compression")
    print("="*50)
    
    test_event = {
        "input": {
            "prompt": "The sky is",
            "model_id": "test-model",
            "batch_size": 48,
            "max_n_logits": 10,
            "desired_logit_prob": 0.95,
            "node_threshold": 0.8,
            "edge_threshold": 0.98,
            "slug_identifier": "test-run-no-compress",
            "max_feature_nodes": 10000,
            "signed_url": "https://httpbin.org/put",
            "user_id": "test-user",
            "compress": False,
            "deadline": None  # No deadline
        }
    }
    
    result = handler(test_event)
    print(f"\nTest Result: {result}")
    return result

def test_token_limit():
    """Test request exceeding token limit"""
    print("\n" + "="*50)
    print("TEST 4: Request exceeding token limit")
    print("="*50)
    
    # Create a very long prompt
    long_prompt = " ".join(["word"] * 200)
    
    test_event = {
        "input": {
            "prompt": long_prompt,
            "model_id": "test-model",
            "batch_size": 48,
            "max_n_logits": 10,
            "desired_logit_prob": 0.95,
            "node_threshold": 0.8,
            "edge_threshold": 0.98,
            "slug_identifier": "test-run",
            "max_feature_nodes": 10000,
            "signed_url": "https://httpbin.org/put",
            "user_id": "test-user",
            "compress": False,
            "deadline": None
        }
    }
    
    result = handler(test_event)
    print(f"\nTest Result: {result}")
    return result

def test_forward_pass():
    """Test forward pass request"""
    print("\n" + "="*50)
    print("TEST 5: Forward pass request")
    print("="*50)
    
    test_event = {
        "input": {
            "prompt": "The capital of France is",
            "max_n_logits": 10,
            "desired_logit_prob": 0.95,
            "request_type": "forward_pass"
        }
    }
    
    result = handler(test_event)
    print(f"\nTest Result: {json.dumps(result, indent=2)}")
    return result

def test_forward_pass_deadline():
    """Test forward pass request with expired deadline"""
    print("\n" + "="*50)
    print("TEST 6: Forward pass with expired deadline")
    print("="*50)
    
    test_event = {
        "input": {
            "prompt": "The capital of France is",
            "max_n_logits": 10,
            "desired_logit_prob": 0.95,
            "deadline": int(time.time()) - 30,  # 30 seconds ago
            "request_type": "forward_pass"
        }
    }
    
    result = handler(test_event)
    print(f"\nTest Result: {json.dumps(result, indent=2)}")
    return result

if __name__ == "__main__":
    print("Running handler tests...")
    
    # Run specific test if provided as argument
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "1":
            test_normal_request()
        elif test_name == "2":
            test_deadline_exceeded()
        elif test_name == "3":
            test_no_compression()
        elif test_name == "4":
            test_token_limit()
        elif test_name == "5":
            test_forward_pass()
        elif test_name == "6":
            test_forward_pass_deadline()
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: 1, 2, 3, 4, 5, 6")
    else:
        # Run all tests
        test_normal_request()
        test_deadline_exceeded()
        test_no_compression()
        test_token_limit()
        test_forward_pass()
        test_forward_pass_deadline()
        
        print("\n" + "="*50)
        print("All tests completed!")
        print("="*50) 