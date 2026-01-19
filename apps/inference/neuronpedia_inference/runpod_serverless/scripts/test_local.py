#!/usr/bin/env python3
"""
Local test script for the RunPod Serverless handler.
Run this to test the handler without RunPod.
"""

import asyncio
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from handler import async_handler, initialize_model


def parse_sse_chunk(chunk: str) -> dict:
    """Parse an SSE-formatted chunk into a dict."""
    if isinstance(chunk, dict):
        return chunk
    # SSE format is "data: {...}\n\n"
    if chunk.startswith("data: "):
        chunk = chunk[6:]
    chunk = chunk.rstrip("\n")
    if chunk:
        return json.loads(chunk)
    return {}


async def test_default_generation():
    """Test simple generation without steering."""
    print("=" * 60)
    print("Test 1: Default generation (no steering)")
    print("=" * 60)
    
    job = {
        "input": {
            "prompt": [
                {"role": "user", "content": "Hello! What is 2 + 2?"}
            ],
            "types": ["DEFAULT"],
            "vectors": [],
            "n_completion_tokens": 100,
            "temperature": 0.7,
        }
    }
    
    print(f"Request: {json.dumps(job, indent=2)}")
    print("\nResponse chunks:")
    
    async for raw_chunk in async_handler(job):
        chunk = parse_sse_chunk(raw_chunk)
        if "error" in chunk:
            print(f"Error: {chunk['error']}")
        else:
            # Print the assistant's response from the last output
            if "outputs" in chunk:
                for output in chunk["outputs"]:
                    chat = output.get("chat_template", [])
                    for msg in chat:
                        if msg.get("role") == "assistant":
                            print(f"Assistant: {msg.get('content', '')[:100]}...")
                            break
            
            if "assistant_axis" in chunk:
                print(f"\nAssistant Axis: {json.dumps(chunk['assistant_axis'], indent=2)}")
    
    print()


async def test_steered_generation():
    """Test generation with steering vector."""
    print("=" * 60)
    print("Test 2: Steered generation")
    print("=" * 60)
    
    # Create a simple mock steering vector (8192 dimensions for Llama 3.3 70B)
    # In real usage, this would be a meaningful steering vector
    mock_vector = [0.001] * 8192
    
    job = {
        "input": {
            "prompt": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a joke."}
            ],
            "types": ["STEERED"],
            "vectors": [
                {
                    "hook": "blocks.40.hook_resid_post",
                    "strength": 1.0,
                    "steering_vector": mock_vector,
                }
            ],
            "strength_multiplier": 1.0,
            "n_completion_tokens": 150,
            "temperature": 0.8,
            "seed": 42,
        }
    }
    
    print(f"Request: prompt with system message and user message")
    print(f"Steering: 1 vector at layer 40")
    print("\nResponse chunks:")
    
    async for raw_chunk in async_handler(job):
        chunk = parse_sse_chunk(raw_chunk)
        if "error" in chunk:
            print(f"Error: {chunk['error']}")
        else:
            if "outputs" in chunk:
                for output in chunk["outputs"]:
                    chat = output.get("chat_template", [])
                    for msg in chat:
                        if msg.get("role") == "assistant":
                            print(f"[{output.get('type', 'UNKNOWN')}] {msg.get('content', '')[:100]}...")
                            break
            
            if "assistant_axis" in chunk:
                print(f"\nAssistant Axis: {json.dumps(chunk['assistant_axis'], indent=2)}")
    
    print()


async def test_both_types():
    """Test generation with both STEERED and DEFAULT."""
    print("=" * 60)
    print("Test 3: Both STEERED and DEFAULT generation")
    print("=" * 60)
    
    mock_vector = [0.001] * 8192
    
    job = {
        "input": {
            "prompt": [
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "types": ["STEERED", "DEFAULT"],
            "vectors": [
                {
                    "hook": "blocks.40.hook_resid_post",
                    "strength": 0.5,
                    "steering_vector": mock_vector,
                }
            ],
            "strength_multiplier": 1.0,
            "n_completion_tokens": 50,
            "temperature": 0.5,
        }
    }
    
    print(f"Request: both STEERED and DEFAULT")
    print("\nFinal responses:")
    
    final_chunk = None
    async for raw_chunk in async_handler(job):
        final_chunk = parse_sse_chunk(raw_chunk)
    
    if final_chunk:
        if "error" in final_chunk:
            print(f"Error: {final_chunk['error']}")
        else:
            if "outputs" in final_chunk:
                for output in final_chunk["outputs"]:
                    print(f"\n[{output.get('type', 'UNKNOWN')}]")
                    chat = output.get("chat_template", [])
                    for msg in chat:
                        if msg.get("role") == "assistant":
                            print(f"  {msg.get('content', '')}")
            
            if "assistant_axis" in final_chunk:
                print(f"\nAssistant Axis data available for {len(final_chunk['assistant_axis'])} type(s)")
    
    print()


async def main():
    """Run all tests."""
    print("Initializing model (this may take a while on first run)...")
    initialize_model()
    print("Model initialized!\n")
    
    # Run tests
    await test_default_generation()
    await test_steered_generation()
    await test_both_types()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

