"""
Completion Chat logic for RunPod Serverless.
Simplified version that only supports VLLMSteerModel with chatspace engine.
"""

import logging
from typing import Any

import torch
from chatspace.generation import VLLMSteerModel
from chatspace.generation.vllm_steer_model import (
    AddSpec,
    LayerSteeringSpec,
    ProjectionCapSpec,
    SteeringSpec,
)
from vllm import SamplingParams

from model import ModelManager
from utils import (
    format_sse_message,
    convert_to_chat_array,
    OrthogonalProjector,
)
from persona_utils.persona_data import PersonaData, DEFAULT_LAYER, ROLE_PC_TITLES
from persona_utils.model_chatspace import ProbingModelChatSpace
from persona_utils.conversation import ConversationEncoder
from persona_utils.spans_chatspace import SpanMapperChatSpace
from persona_utils.analysis import pc_projection

logger = logging.getLogger(__name__)

# System prompt addition for assistant_axis mode
_ASSISTANT_AXIS_SYSTEM_PROMPT_ADDITION = """Keep your responses short and to the point, ideally less than 3 sentences unless asked otherwise."""


async def run_completion_chat_streaming(
    model_manager: ModelManager,
    prompt: list[dict],
    steer_types: list[str],
    vectors: list[dict],
    strength_multiplier: float = 1.0,
    seed: int | None = None,
    temperature: float = 0.7,
    freq_penalty: float = 0.0,
    n_completion_tokens: int = 512,
    steer_method: str = "SIMPLE_ADDITIVE",
    normalize_steering: bool = False,
    steer_special_tokens: bool = False,
):
    """
    Run streaming completion chat generation.
    
    This is a simplified version that only supports VLLMSteerModel.
    Always runs with is_assistant_axis=True for persona monitoring.
    
    Args:
        model_manager: The ModelManager instance
        prompt: List of chat messages [{"role": "user", "content": "..."}]
        steer_types: List of steer types ["STEERED", "DEFAULT"]
        vectors: List of steering vectors
        strength_multiplier: Strength multiplier for steering
        seed: Random seed
        temperature: Sampling temperature
        freq_penalty: Frequency penalty (ignored for VLLMSteerModel)
        n_completion_tokens: Maximum tokens to generate
        steer_method: Steering method (SIMPLE_ADDITIVE or PROJECTION_CAP)
        normalize_steering: Whether to normalize steering vectors
        steer_special_tokens: Whether to steer special tokens
    
    Yields:
        SSE-formatted response chunks
    """
    model = model_manager.get_model()
    
    # Validate steer types order (STEERED must come before DEFAULT)
    if "STEERED" in steer_types and "DEFAULT" in steer_types:
        if steer_types.index("STEERED") > steer_types.index("DEFAULT"):
            yield format_sse_message({"error": "STEERED must come before DEFAULT"})
            return
    
    # Format prompt with assistant_axis system prompt
    prompt_formatted = _format_prompt_with_assistant_axis(prompt)
    
    # Apply chat template
    prompt_tokenized = model.tokenizer.apply_chat_template(
        prompt_formatted, tokenize=True, add_generation_prompt=True
    )
    
    # Remove BOS if present (vLLM adds it)
    if prompt_tokenized[0] == model.tokenizer.bos_token_id:
        prompt_tokenized = prompt_tokenized[1:]
    
    prompt_tokenized = torch.tensor(prompt_tokenized)
    prompt_string = model.tokenizer.decode(prompt_tokenized)
    
    # Check if generating both types
    generate_both = "STEERED" in steer_types and "DEFAULT" in steer_types
    
    if generate_both:
        async for chunk in _generate_both_types(
            model=model,
            prompt_string=prompt_string,
            prompt_tokenized=prompt_tokenized,
            prompt_formatted=prompt_formatted,
            steer_types=steer_types,
            vectors=vectors,
            strength_multiplier=strength_multiplier,
            seed=seed,
            temperature=temperature,
            n_completion_tokens=n_completion_tokens,
            steer_method=steer_method,
            normalize_steering=normalize_steering,
        ):
            yield chunk
    else:
        steer_type = steer_types[0]
        async for chunk in _generate_single_type(
            model=model,
            prompt_string=prompt_string,
            prompt_tokenized=prompt_tokenized,
            prompt_formatted=prompt_formatted,
            steer_type=steer_type,
            vectors=vectors,
            strength_multiplier=strength_multiplier,
            seed=seed,
            temperature=temperature,
            n_completion_tokens=n_completion_tokens,
            steer_method=steer_method,
            normalize_steering=normalize_steering,
        ):
            yield chunk


def _format_prompt_with_assistant_axis(prompt: list[dict]) -> list[dict]:
    """Add assistant_axis system prompt to the conversation."""
    addition_text = _ASSISTANT_AXIS_SYSTEM_PROMPT_ADDITION.strip()
    prompt_formatted = []
    
    if prompt and prompt[0].get("role") == "system":
        # Existing system message - add our addition
        system_content = prompt[0].get("content", "")
        # Remove any existing additions (dedup)
        while addition_text in system_content:
            system_content = system_content.replace(addition_text, "")
        system_content = system_content.strip()
        # Re-add at beginning
        if system_content:
            system_content = addition_text + "\n" + system_content
        else:
            system_content = addition_text
        prompt_formatted.append({"role": "system", "content": system_content})
        prompt_formatted.extend(prompt[1:])
    else:
        # No system message - prepend one
        prompt_formatted.append({"role": "system", "content": addition_text})
        prompt_formatted.extend(prompt)
    
    return prompt_formatted


def _build_steering_spec(
    vectors: list[dict],
    strength_multiplier: float,
    steer_method: str,
    normalize_steering: bool,
) -> SteeringSpec:
    """Build a SteeringSpec from vectors."""
    # Group vectors by layer
    layer_features: dict[int, list[tuple[dict, torch.Tensor]]] = {}
    
    for vector in vectors:
        hook_name = vector.get("hook", "")
        if "resid_post" in hook_name:
            layer = int(hook_name.split(".")[1])
        elif "resid_pre" in hook_name:
            layer = int(hook_name.split(".")[1]) - 1
        else:
            raise ValueError(f"Unsupported hook name: {hook_name}")
        
        steering_vector = torch.tensor(vector.get("steering_vector", []))
        
        if not torch.isfinite(steering_vector).all():
            raise ValueError("Steering vector contains inf or nan values")
        
        if normalize_steering:
            norm = torch.norm(steering_vector)
            if norm == 0:
                raise ValueError("Zero norm steering vector")
            steering_vector = steering_vector / norm
        
        if layer not in layer_features:
            layer_features[layer] = []
        layer_features[layer].append((vector, steering_vector))
    
    # Build LayerSteeringSpec for each layer
    steering_spec_layers = {}
    
    for layer, features in layer_features.items():
        operations = []
        
        if steer_method == "SIMPLE_ADDITIVE":
            for vector, steering_vector in features:
                coeff = strength_multiplier * vector.get("strength", 1.0)
                norm = torch.norm(steering_vector)
                if norm > 0:
                    normalized_vector = steering_vector / norm
                    operations.append(AddSpec(
                        vector=normalized_vector,
                        scale=norm.item() * coeff,
                    ))
        
        elif steer_method == "PROJECTION_CAP":
            for vector, steering_vector in features:
                coeff = strength_multiplier * vector.get("strength", 1.0)
                operations.append(ProjectionCapSpec(
                    vector=steering_vector,
                    min=None,
                    max=coeff,
                ))
        
        else:
            raise ValueError(f"Unsupported steer method: {steer_method}")
        
        if operations:
            steering_spec_layers[layer] = LayerSteeringSpec(operations=operations)
    
    if not steering_spec_layers:
        raise ValueError("No valid steering layers found")
    
    return SteeringSpec(layers=steering_spec_layers)


def _make_response(
    steer_types: list[str],
    steered_result: str,
    default_result: str,
    prompt_formatted: list[dict],
    tokenizer: Any,
    assistant_axis_data: list[dict] | None = None,
) -> dict:
    """Build the response dictionary."""
    outputs = []
    
    for steer_type in steer_types:
        if steer_type == "STEERED":
            outputs.append({
                "raw": steered_result,
                "chat_template": convert_to_chat_array(steered_result, tokenizer),
                "type": steer_type,
            })
        else:
            outputs.append({
                "raw": default_result,
                "chat_template": convert_to_chat_array(default_result, tokenizer),
                "type": steer_type,
            })
    
    response = {
        "outputs": outputs,
        "input": {
            "chat_template": prompt_formatted,
        },
    }
    
    if assistant_axis_data:
        response["assistant_axis"] = assistant_axis_data
    
    return response


async def _generate_single_type(
    model: VLLMSteerModel,
    prompt_string: str,
    prompt_tokenized: torch.Tensor,
    prompt_formatted: list[dict],
    steer_type: str,
    vectors: list[dict],
    strength_multiplier: float,
    seed: int | None,
    temperature: float,
    n_completion_tokens: int,
    steer_method: str,
    normalize_steering: bool,
):
    """Generate a single type (STEERED or DEFAULT)."""
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=n_completion_tokens,
        seed=seed,
    )
    
    # Store steering_spec for persona monitor
    steering_spec = None
    
    if steer_type == "STEERED":
        steering_spec = _build_steering_spec(
            vectors, strength_multiplier, steer_method, normalize_steering
        )
        stream_generator = await model.generate(
            prompt_string,
            sampling_params,
            steering_spec=steering_spec,
            stream=True,
        )
    else:
        stream_generator = await model.generate(
            prompt_string,
            sampling_params,
            stream=True,
        )
    
    output_total = ""
    async for delta in stream_generator:
        output_total += delta
        response = _make_response(
            [steer_type],
            prompt_string + output_total,
            prompt_string + output_total,
            prompt_formatted,
            model.tokenizer,
        )
        yield format_sse_message(response)
    
    # Run persona monitor after generation completes
    # Pass steering_spec for STEERED type to get post-cap values
    full_conversation = prompt_formatted + [{"role": "assistant", "content": output_total}]
    axis_data = await _run_persona_monitor(
        model, full_conversation, steer_type, DEFAULT_LAYER,
        steering_spec=steering_spec if steer_type == "STEERED" else None
    )
    
    response = _make_response(
        [steer_type],
        prompt_string + output_total,
        prompt_string + output_total,
        prompt_formatted,
        model.tokenizer,
        [axis_data] if axis_data else None,
    )
    yield format_sse_message(response)


async def _generate_both_types(
    model: VLLMSteerModel,
    prompt_string: str,
    prompt_tokenized: torch.Tensor,
    prompt_formatted: list[dict],
    steer_types: list[str],
    vectors: list[dict],
    strength_multiplier: float,
    seed: int | None,
    temperature: float,
    n_completion_tokens: int,
    steer_method: str,
    normalize_steering: bool,
):
    """Generate both STEERED and DEFAULT types."""
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=n_completion_tokens,
        seed=seed,
    )
    
    steered_output = ""
    default_output = ""
    
    # Generate STEERED first
    steering_spec = _build_steering_spec(
        vectors, strength_multiplier, steer_method, normalize_steering
    )
    stream_generator = await model.generate(
        prompt_string,
        sampling_params,
        steering_spec=steering_spec,
        stream=True,
    )
    
    async for delta in stream_generator:
        steered_output += delta
        response = _make_response(
            steer_types,
            prompt_string + steered_output,
            prompt_string + default_output,
            prompt_formatted,
            model.tokenizer,
        )
        yield format_sse_message(response)
    
    # Generate DEFAULT
    stream_generator = await model.generate(
        prompt_string,
        sampling_params,
        stream=True,
    )
    
    async for delta in stream_generator:
        default_output += delta
        response = _make_response(
            steer_types,
            prompt_string + steered_output,
            prompt_string + default_output,
            prompt_formatted,
            model.tokenizer,
        )
        yield format_sse_message(response)
    
    # Run persona monitor for both types
    assistant_axis_data_list = []
    
    for steer_type in steer_types:
        if steer_type == "STEERED":
            output_for_monitor = steered_output
        else:
            output_for_monitor = default_output
        
        full_conversation = prompt_formatted + [{"role": "assistant", "content": output_for_monitor}]
        # Pass steering_spec for STEERED type to get post-cap values
        axis_data = await _run_persona_monitor(
            model, full_conversation, steer_type, DEFAULT_LAYER,
            steering_spec=steering_spec if steer_type == "STEERED" else None
        )
        if axis_data:
            assistant_axis_data_list.append(axis_data)
    
    # Yield final response with persona data
    response = _make_response(
        steer_types,
        prompt_string + steered_output,
        prompt_string + default_output,
        prompt_formatted,
        model.tokenizer,
        assistant_axis_data_list if assistant_axis_data_list else None,
    )
    yield format_sse_message(response)


async def _run_persona_monitor(
    model: VLLMSteerModel,
    full_conversation: list[dict],
    steer_type: str,
    layer: int,
    steering_spec: Any = None,
) -> dict | None:
    """
    Run persona monitoring on the conversation.
    
    Args:
        model: VLLMSteerModel instance
        full_conversation: List of message dicts with role and content
        steer_type: The steer type this analysis corresponds to
        layer: Layer to extract activations from
        steering_spec: Optional SteeringSpec to apply during capture. If provided,
            both pre-cap (base model) and post-cap (with steering) activations are captured.
    
    Returns assistant_axis data for the given steer type.
    """
    persona_data = PersonaData.get_instance()
    if not persona_data.is_initialized():
        logger.warning("Persona data not initialized, skipping persona monitor")
        return None
    
    pca_results = persona_data.get_pca_data(layer)
    if pca_results is None:
        logger.warning(f"PCA data not available for layer {layer}")
        return None
    
    # Get model ID for persona data
    model_id = persona_data.model_id
    
    # Wrap model with ProbingModelChatSpace
    probing_model = ProbingModelChatSpace.from_existing(
        model,
        tokenizer=None,
        model_name=model_id,
    )
    
    tokenizer = probing_model.tokenizer
    encoder = ConversationEncoder(tokenizer, model_id)
    mapper = SpanMapperChatSpace(tokenizer)
    
    # Extract mean activations per turn (pre-cap / base model)
    logger.info("Extracting pre-cap activations for persona monitor")
    mean_acts_per_turn = await mapper.mean_all_turn_activations_async(
        probing_model, encoder, full_conversation, layer=layer
    )
    
    # Extract post-cap activations if steering_spec is provided
    mean_acts_per_turn_post_cap = None
    if steering_spec is not None:
        logger.info("Extracting post-cap activations with steering_spec")
        mean_acts_per_turn_post_cap = await mapper.mean_all_turn_activations_async(
            probing_model, encoder, full_conversation, layer=layer, steering_spec=steering_spec
        )
    
    # Handle empty activations
    if mean_acts_per_turn.shape[0] == 0:
        logger.warning("No activations extracted, skipping persona monitor")
        return None
    
    # Compute projections (pre-cap)
    role_projs = pc_projection(mean_acts_per_turn, pca_results, n_pcs=1)
    
    # Compute projections (post-cap) if available
    role_projs_post_cap = None
    if mean_acts_per_turn_post_cap is not None and mean_acts_per_turn_post_cap.shape[0] > 0:
        role_projs_post_cap = pc_projection(mean_acts_per_turn_post_cap, pca_results, n_pcs=1)
    
    # Find indices of assistant turns
    assistant_indices = [i for i, msg in enumerate(full_conversation) if msg["role"] == "assistant"]
    assistant_role_projs = role_projs[assistant_indices] if assistant_indices else role_projs[0:0]
    assistant_role_projs_post_cap = None
    if role_projs_post_cap is not None:
        assistant_role_projs_post_cap = role_projs_post_cap[assistant_indices] if assistant_indices else role_projs_post_cap[0:0]
    
    # Get assistant turns for snippets
    assistant_turns = [msg for msg in full_conversation if msg["role"] == "assistant"]
    
    turns_data = []
    for i in range(len(assistant_role_projs)):
        pc_values = {
            ROLE_PC_TITLES[j]: float(assistant_role_projs[i][j])
            for j in range(len(ROLE_PC_TITLES))
        }
        
        # Add post-cap values if available
        pc_values_post_cap = None
        if assistant_role_projs_post_cap is not None and i < len(assistant_role_projs_post_cap):
            pc_values_post_cap = {
                ROLE_PC_TITLES[j]: float(assistant_role_projs_post_cap[i][j])
                for j in range(len(ROLE_PC_TITLES))
            }
        
        snippet = ""
        if i < len(assistant_turns):
            content = assistant_turns[i]["content"]
            snippet = content[:120] + "..." if len(content) > 120 else content
        
        turn_data = {
            "pc_values": pc_values,
            "snippet": snippet,
        }
        if pc_values_post_cap is not None:
            turn_data["pc_values_post_cap"] = pc_values_post_cap
        
        turns_data.append(turn_data)
    
    return {
        "type": steer_type,
        "pc_titles": list(ROLE_PC_TITLES),
        "turns": turns_data,
    }

