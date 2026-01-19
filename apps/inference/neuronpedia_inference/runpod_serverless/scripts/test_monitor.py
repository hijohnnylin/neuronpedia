#!/usr/bin/env python
"""
Test script for the vLLM health monitor.

Run this to see current system stats even without a model loaded.

Usage:
    cd /root/np-cap/apps/inference/neuronpedia_inference/runpod_serverless/src
    python ../scripts/test_monitor.py
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vllm_monitor import VLLMMonitor, get_health_stats


async def main():
    print("Testing vLLM Health Monitor (no model loaded)\n")
    
    # Create monitor without model
    monitor = VLLMMonitor()
    
    # Get stats
    stats = await monitor.get_stats()
    
    # Print human-readable summary
    print(stats.summary())
    print()
    
    # Print raw dict for debugging
    print("Raw stats dict:")
    import json
    print(json.dumps(stats.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())

