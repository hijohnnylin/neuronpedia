"""
Model management for RunPod Serverless.
Handles loading and managing the VLLMSteerModel.
"""

import logging
import os
from typing import Any

import torch
from chatspace.generation import VLLMSteerModel, VLLMSteeringConfig

logger = logging.getLogger(__name__)

STR_TO_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class ModelManager:
    """Manages the VLLMSteerModel instance."""
    
    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.3-70B-Instruct",
        override_model_id: str | None = "casperhansen/llama-3.3-70b-instruct-awq",
        model_dtype: str = "bfloat16",
        token_limit: int = 65536,
    ):
        self.model_id = model_id
        self.override_model_id = override_model_id or model_id
        self.model_dtype = model_dtype
        self.token_limit = token_limit
        self.model: VLLMSteerModel | None = None
    
    def load_model(self) -> None:
        """Load the VLLMSteerModel."""
        if self.model is not None:
            logger.info("Model already loaded")
            return
        
        logger.info(f"Loading VLLMSteerModel: {self.override_model_id}")
        
        # Get max model length from environment or use token_limit
        max_model_len = int(os.environ.get("MAX_MODEL_LEN", str(self.token_limit)))
        
        # Get tensor parallel size from environment (default 1)
        tensor_parallel_size = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
        
        # Get GPU memory utilization from environment
        gpu_memory_utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.95"))
        
        # Get max concurrent sequences from environment (enables multiple requests per GPU)
        max_num_seqs = int(os.environ.get("MAX_NUM_SEQS", "32"))
        
        config = VLLMSteeringConfig(
            model_name=self.override_model_id,
            dtype=self.model_dtype,  # Pass as string, e.g. "bfloat16"
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        
        self.model = VLLMSteerModel(
            config,
            bootstrap_layers=(),
            enable_prefix_caching=False,  # Ensures old caches don't "bleed" over to new requests
            enable_chunked_prefill=False,  # Required for full activation capture
            max_num_batched_tokens=max_model_len,  # Match token limit for batching
            max_num_seqs=max_num_seqs,  # Allow multiple concurrent sequences per GPU
        )
        
        logger.info(f"Model loaded successfully. Hidden size: {self.model.hidden_size}, Layers: {self.model.layer_count}")
    
    def get_model(self) -> VLLMSteerModel:
        """Get the loaded model instance."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self.model
    
    @property
    def tokenizer(self):
        """Get the tokenizer from the model."""
        return self.get_model().tokenizer

