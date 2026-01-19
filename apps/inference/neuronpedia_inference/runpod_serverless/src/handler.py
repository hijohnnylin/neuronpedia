"""
RunPod Serverless Handler for Neuronpedia Inference
Specifically for: Llama 3.3 70B AWQ with ChatSpace engine, completion_chat endpoint, assistant_axis=true
"""

import logging
import os
from typing import Any

import torch

# RunPod import is optional for local testing
try:
    import runpod
    RUNPOD_AVAILABLE = True
except ImportError:
    RUNPOD_AVAILABLE = False

from model import ModelManager
from completion_chat import run_completion_chat_streaming
from persona_utils.persona_data import PersonaData, initialize_persona_data, DEFAULT_LAYER
from vllm_monitor import VLLMMonitor, get_monitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable background health monitoring if env var is set
ENABLE_BACKGROUND_MONITOR = os.environ.get("ENABLE_VLLM_MONITOR", "0") == "1"
MONITOR_INTERVAL = float(os.environ.get("VLLM_MONITOR_INTERVAL", "30"))

# Global model manager (initialized once at cold start)
model_manager: ModelManager | None = None


def initialize_model():
    """Initialize the model at cold start."""
    global model_manager
    
    if model_manager is not None:
        logger.info("Model already initialized")
        return
    
    logger.info("Initializing model...")
    
    # Configuration from environment or defaults
    model_id = os.environ.get("MODEL_ID", "meta-llama/Llama-3.3-70B-Instruct")
    override_model_id = os.environ.get("OVERRIDE_MODEL_ID", "casperhansen/llama-3.3-70b-instruct-awq")
    model_dtype = os.environ.get("MODEL_DTYPE", "bfloat16")
    token_limit = int(os.environ.get("TOKEN_LIMIT", "65536"))
    
    model_manager = ModelManager(
        model_id=model_id,
        override_model_id=override_model_id,
        model_dtype=model_dtype,
        token_limit=token_limit,
    )
    
    model_manager.load_model()
    
    # Initialize persona data for assistant_axis
    logger.info("Initializing persona data...")
    initialize_persona_data(override_model_id, layers=[DEFAULT_LAYER])
    
    # Set up health monitor
    monitor = get_monitor()
    monitor.set_model_manager(model_manager)
    
    logger.info("Model and persona data initialized successfully")


async def async_handler(job: dict) -> dict:
    """
    Async handler for the completion_chat endpoint.
    
    Expected input format:
    {
        "input": {
            "prompt": [{"role": "user", "content": "Hello"}, ...],
            "types": ["STEERED"],  # or ["DEFAULT"] or ["STEERED", "DEFAULT"]
            "vectors": [{"hook": "blocks.40.hook_resid_post", "strength": 1.0, "steering_vector": [...]}],
            "strength_multiplier": 1.0,
            "seed": 42,
            "temperature": 0.7,
            "freq_penalty": 0.0,
            "n_completion_tokens": 512,
            "steer_method": "SIMPLE_ADDITIVE",
            "normalize_steering": false,
            "steer_special_tokens": false,
            "action": "generate",  # Optional: "generate" (default) or "health"
        }
    }
    
    Returns streaming SSE response via yield.
    """
    import time
    global model_manager
    
    input_data = job.get("input", {})
    action = input_data.get("action", "generate")
    
    # Handle health check action
    if action == "health":
        monitor = get_monitor()
        if model_manager is not None:
            monitor.set_model_manager(model_manager)
        stats = await monitor.get_stats()
        yield {
            "action": "health",
            "stats": stats.to_dict(),
            "summary": stats.summary(),
        }
        return
    
    request_start = time.time()
    logger.info(f"[REQUEST START] action={action}")
    
    if model_manager is None:
        initialize_model()
    
    # Start background monitoring if enabled (once)
    if ENABLE_BACKGROUND_MONITOR:
        monitor = get_monitor()
        monitor.set_model_manager(model_manager)
        if monitor._background_task is None:
            monitor.start_background_logging(interval=MONITOR_INTERVAL)
    
    # Ensure vLLM async engine is initialized (happens once, within RunPod's event loop)
    logger.info("[REQUEST] Ensuring engine initialized...")
    engine_init_start = time.time()
    await model_manager.get_model()._inner._ensure_engine_initialized()
    logger.info(f"[REQUEST] Engine ready (took {time.time() - engine_init_start:.2f}s)")
    
    # Extract request parameters
    prompt = input_data.get("prompt", [])
    steer_types = input_data.get("types", ["DEFAULT"])
    vectors = input_data.get("vectors", [])
    strength_multiplier = float(input_data.get("strength_multiplier", 1.0))
    seed = input_data.get("seed")
    if seed is not None:
        seed = int(seed)
    temperature = float(input_data.get("temperature", 0.7))
    freq_penalty = float(input_data.get("freq_penalty", 0.0))
    n_completion_tokens = int(input_data.get("n_completion_tokens", 512))
    steer_method = input_data.get("steer_method", "SIMPLE_ADDITIVE")
    normalize_steering = input_data.get("normalize_steering", False)
    steer_special_tokens = input_data.get("steer_special_tokens", False)
    
    # Validate input
    if not prompt:
        yield {"error": "prompt is required"}
        return
    
    if not vectors and "STEERED" in steer_types:
        yield {"error": "vectors are required when STEERED type is requested"}
        return
    
    # Run the streaming generation
    generation_start = time.time()
    token_count = 0
    try:
        logger.info(f"[REQUEST] Starting generation: types={steer_types}, max_tokens={n_completion_tokens}")
        async for chunk in run_completion_chat_streaming(
            model_manager=model_manager,
            prompt=prompt,
            steer_types=steer_types,
            vectors=vectors,
            strength_multiplier=strength_multiplier,
            seed=seed,
            temperature=temperature,
            freq_penalty=freq_penalty,
            n_completion_tokens=n_completion_tokens,
            steer_method=steer_method,
            normalize_steering=normalize_steering,
            steer_special_tokens=steer_special_tokens,
        ):
            token_count += 1
            yield chunk
        
        generation_time = time.time() - generation_start
        total_time = time.time() - request_start
        logger.info(
            f"[REQUEST COMPLETE] total={total_time:.2f}s, generation={generation_time:.2f}s, "
            f"~tokens={token_count}, tokens/sec={token_count/generation_time:.1f}"
        )
    except Exception as e:
        logger.exception(f"[REQUEST ERROR] Error during generation after {time.time() - request_start:.2f}s")
        yield {"error": str(e)}


# Concurrency modifier function for RunPod
# Returns how many jobs this worker can handle concurrently
def concurrency_modifier(current_concurrency: int) -> int:
    """
    Called by RunPod to determine how many concurrent jobs this worker can handle.
    
    This enables running multiple requests per GPU by utilizing vLLM's continuous batching.
    vLLM handles the actual batching/scheduling - we just need to accept multiple jobs.
    
    Args:
        current_concurrency: The number of jobs currently being processed
    
    Returns:
        Maximum number of concurrent jobs to allow
    """
    # Get max concurrency from environment, default to 32 to match MAX_NUM_SEQS
    max_concurrency = int(os.environ.get("MAX_CONCURRENCY", "32"))
    return max_concurrency


# Only run when executed directly (not when imported for testing)
if __name__ == "__main__":
    # Initialize model at module load (cold start)
    initialize_model()
    # Register handler with RunPod
    # RunPod supports async generators directly - no need to wrap
    if RUNPOD_AVAILABLE:
        runpod.serverless.start({
            "handler": async_handler,  # Pass async generator directly
            "return_aggregate_stream": True,
            "concurrency_modifier": concurrency_modifier,  # Enable concurrent job handling
        })
    else:
        print("RunPod not available. Use test_local.py for testing.")

