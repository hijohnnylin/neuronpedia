"""Persona monitoring endpoint for the inference server.

This endpoint uses the existing ChatSpace/vLLM model weights to analyze
persona characteristics in conversations without loading a second copy.
"""

import logging

import numpy as np
import torch
from chatspace.generation import VLLMSteerModel
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from neuronpedia_inference.config import Config
from neuronpedia_inference.shared import Model

from .utils import (
    ConversationEncoder,
    ProbingModelChatSpace,
    SpanMapperChatSpace,
    PersonaData,
    DEFAULT_LAYER,
    ROLE_PC_TITLES,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# =============================================================================
# Allowed Models Whitelist (exact match required)
# =============================================================================
ALLOWED_MODELS = {
    "meta-llama/Llama-3.3-70B-Instruct",
    "casperhansen/llama-3.3-70b-instruct-awq",
}


# =============================================================================
# Request/Response Models (defined here for visibility, no schema generation)
# =============================================================================
class ConversationTurn(BaseModel):
    """A single turn in a conversation."""
    role: str
    content: str


class PersonaMonitorRequest(BaseModel):
    """
    Request format for persona monitoring.
    
    Attributes:
        model: Model identifier (must be in ALLOWED_MODELS whitelist)
        layer: Layer to extract activations from (default: 40)
        conversation: List of conversation turns with role and content
    """
    model: str
    layer: int = DEFAULT_LAYER
    conversation: list[ConversationTurn]


class TurnData(BaseModel):
    """
    Data for a single turn in the response.
    
    Attributes:
        pc_values: Dict mapping PC title to projection value
        snippet: Truncated conversation content for this turn
    """
    pc_values: dict[str, float]
    snippet: str


class PersonaMonitorResponse(BaseModel):
    """
    Response format for persona monitoring.
    
    Attributes:
        pc_titles: List of principal component titles/descriptions
        turns: List of turn data with PC values and conversation snippets
    """
    pc_titles: list[str]
    turns: list[TurnData]



# =============================================================================
# Analysis Functions
# =============================================================================
def pc_projection(mean_acts_per_turn: torch.Tensor, pca_results: dict, n_pcs: int = 1) -> np.ndarray:
    """
    Project activations onto principal components.
    
    Args:
        mean_acts_per_turn: Tensor of shape (num_turns, hidden_size)
        pca_results: Dict with 'pca' and 'scaler'
        n_pcs: Number of principal components to project onto
        
    Returns:
        Array of shape (num_turns, n_pcs) with projection values
    """
    if isinstance(mean_acts_per_turn, list):
        stacked_acts = torch.stack(mean_acts_per_turn)
    else:
        stacked_acts = mean_acts_per_turn
    
    stacked_acts = stacked_acts.float().numpy()
    scaled_acts = pca_results["scaler"].transform(stacked_acts)
    projected_acts = pca_results["pca"].transform(scaled_acts)
    
    return projected_acts[:, :n_pcs]


def _truncate_content(content: str, max_length: int = 120) -> str:
    """Truncate content to a reasonable length for snippets."""
    if len(content) <= max_length:
        return content
    return content[:max_length] + "..."


def _is_model_allowed(model_id: str) -> bool:
    """Check if model ID is in the allowed list (exact match)."""
    return model_id in ALLOWED_MODELS


# =============================================================================
# Endpoint
# =============================================================================
@router.post("/persona/monitor")
async def persona_monitor(request: PersonaMonitorRequest) -> PersonaMonitorResponse:
    """
    Monitor persona characteristics in a conversation.
    
    This endpoint extracts hidden state activations from the conversation
    and projects them onto pre-computed principal components that capture
    persona-related variation in the model's representations.
    
    The endpoint uses the existing loaded ChatSpace/vLLM model weights,
    wrapping them with ProbingModelChatSpace.from_existing() to avoid
    loading duplicate model weights.
    
    Returns:
        PersonaMonitorResponse with PC titles and per-turn projection values
        including conversation snippets.
    """
    logger.info(f"Received persona-monitor request with {len(request.conversation)} messages")
    logger.info(f"Model: {request.model}, Layer: {request.layer}")
    
    # Validate model against whitelist (exact match)
    if not _is_model_allowed(request.model):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' is not in the allowed models list. "
                   f"Allowed models: {list(ALLOWED_MODELS)}"
        )
    
    # Get the existing model instance
    model = Model.get_instance()
    
    if not isinstance(model, VLLMSteerModel):
        raise HTTPException(
            status_code=500,
            detail="Persona monitoring requires ChatSpace/vLLM backend. "
                   "The inference server must be started with --chatspace flag."
        )
    
    # Get model name from config for data path lookup
    config = Config.get_instance()
    model_id_for_data = config.override_model_id or config.model_id
    
    # Get pre-loaded PCA data from PersonaData singleton
    persona_data = PersonaData.get_instance()
    if not persona_data.is_initialized():
        raise HTTPException(
            status_code=500,
            detail="Persona data not initialized. Server may not have started with chatspace mode."
        )
    
    pca_results = persona_data.get_pca_data(request.layer)
    if pca_results is None:
        raise HTTPException(
            status_code=400,
            detail=f"PCA data not available for layer {request.layer}. "
                   f"Data may not have been loaded at server startup."
        )
    
    # Wrap the existing model with ProbingModelChatSpace
    probing_model = ProbingModelChatSpace.from_existing(
        model, 
        tokenizer=None,  # VLLMSteerModel has its own tokenizer
        model_name=model_id_for_data
    )
    
    tokenizer = probing_model.tokenizer
    encoder = ConversationEncoder(tokenizer, model_id_for_data)
    mapper = SpanMapperChatSpace(tokenizer)
    
    # Extract mean activations per turn
    logger.info("Extracting activations using ChatSpace backend")
    mean_acts_per_turn = await mapper.mean_all_turn_activations_async(
        probing_model, encoder, request.conversation, layer=request.layer
    )
    
    logger.info(f"Activations shape: {mean_acts_per_turn.shape}")
    
    # Handle empty activations
    if mean_acts_per_turn.shape[0] == 0:
        raise HTTPException(
            status_code=500,
            detail="No activations could be extracted from the conversation. "
                   "This may be due to a tokenization mismatch or context length issue."
        )
    
    # Compute projections
    role_projs = pc_projection(mean_acts_per_turn, pca_results, n_pcs=1)
    logger.info(f"Projections shape: {role_projs.shape}")
    
    # Build response with assistant turns only (every other turn starting from index 1)
    # Index 0 = user, index 1 = assistant, index 2 = user, index 3 = assistant, etc.
    assistant_role_projs = role_projs[1::2]
    
    # Get assistant turns from the conversation for snippets
    assistant_turns = [
        turn for turn in request.conversation if turn.role == "assistant"
    ]
    
    turns_data = []
    for i in range(len(assistant_role_projs)):
        pc_values = {
            ROLE_PC_TITLES[j]: float(assistant_role_projs[i][j])
            for j in range(len(ROLE_PC_TITLES))
        }
        
        # Get snippet from corresponding assistant turn
        snippet = ""
        if i < len(assistant_turns):
            snippet = _truncate_content(assistant_turns[i].content)
        
        turns_data.append(TurnData(pc_values=pc_values, snippet=snippet))
    
    return PersonaMonitorResponse(
        pc_titles=list(ROLE_PC_TITLES),
        turns=turns_data
    )

