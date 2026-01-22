import re
from collections import defaultdict

import torch
from neuronpedia_inference_client.models.np_steer_chat_message import (
    NPSteerChatMessage,
)
from neuronpedia_inference_client.models.np_steer_feature import NPSteerFeature
from transformers import PreTrainedTokenizerBase

from neuronpedia_inference.config import Config
from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.shared import request_lock


# Regex to match Llama 3's auto-injected knowledge cutoff preamble in system messages
# This preamble is added by apply_chat_template and looks like:
#   Cutting Knowledge Date: December 2023
#   Today Date: 26 Jul 2024
#
# We strip it to prevent duplication when the conversation is sent back and
# apply_chat_template is called again
_LLAMA3_SYSTEM_PREAMBLE_PATTERN = re.compile(
    r"^Cutting Knowledge Date:\s*[^\n]+\nToday Date:\s*[^\n]+\n*",
    re.MULTILINE
)

# no additional system prompt addition for assistant axis
# _ASSISTANT_AXIS_SYSTEM_PROMPT_ADDITION = ""

def _strip_llama3_system_preamble(content: str) -> str:
    """Strip Llama 3's auto-injected knowledge cutoff preamble and assistant axis system prompt addition from system message content."""
    content = _LLAMA3_SYSTEM_PREAMBLE_PATTERN.sub("", content)
    return content.strip()


async def stream_lock(is_stream: bool):
    if is_stream:
        return request_lock

    class DummyLock:
        async def __aenter__(self):
            pass

        async def __aexit__(self, *args):  # type: ignore
            pass

    return DummyLock()


def format_sse_message(data: str) -> str:
    return f"data: {data}\n\n"


def remove_sse_formatting(data: str) -> str:
    if data.startswith("data: "):
        data = data[6:]  # Remove "data: " prefix
    return data.rstrip("\n\n")


def process_features_vectorized(features: list[NPSteerFeature]):
    # Group features by source
    source_groups: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    for i, feature in enumerate(features):
        source_groups[feature.source].append((i, int(feature.index)))

    # Process by each source
    for source, indices in source_groups.items():
        sae = SAEManager.get_instance().get_sae(source)
        feature_indices = torch.tensor(
            [idx for _, idx in indices], device=sae.W_dec.device
        )
        steering_vectors = sae.W_dec[feature_indices]

        # Assign steering vectors back to features
        for (feature_idx, _), steer_vector in zip(indices, steering_vectors):
            features[feature_idx].steering_vector = steer_vector

    return features


# TODO: We should have a more generic way to handle this
def convert_to_chat_array(
    text: str,
    tokenizer: PreTrainedTokenizerBase | None,
    custom_hf_model_id: str | None = None,
) -> list[NPSteerChatMessage]:
    config = Config.get_instance()
    if tokenizer is None:
        # Handle the None case
        # Either raise an error:
        raise ValueError("Tokenizer cannot be None for chat array conversion")
    # Tokenize the input text
    tokens = tokenizer.encode(text)

    # Initialize variables
    conversation: list[NPSteerChatMessage] = []
    current_role = None
    current_content = []

    # case: gpt-oss-20b (harmony chat format)
    # Format: <|start|>role<|message|>content<|end|> or <|start|>role<|channel|>channel_name<|message|>content<|end|>
    if hasattr(tokenizer, "name_or_path") and "gpt-oss" in tokenizer.name_or_path:
        # Split by <|start|> to get conversation turns
        parts = text.split("<|start|>")
        # Store pending analysis content to merge with final channel
        pending_analysis: str | None = None

        for part in parts[1:]:  # Skip first empty part
            if not part.strip():
                continue

            # Extract content up to <|end|> or <|return|> if present, otherwise use the whole part
            if "<|end|>" in part:
                content_part = part.split("<|end|>")[0]
            elif "<|return|>" in part:
                content_part = part.split("<|return|>")[0]
            else:
                # Handle last message without end marker (still being generated)
                content_part = part

            # Extract role and channel (text before <|channel|> or <|message|>)
            channel = None
            if "<|channel|>" in content_part:
                role = content_part.split("<|channel|>")[0].strip()
                # Get the part after <|channel|> to find channel name and message
                after_channel = content_part.split("<|channel|>")[1]
                if "<|message|>" in after_channel:
                    channel = after_channel.split("<|message|>")[0].strip()
                    content = after_channel.split("<|message|>")[1].strip()
                else:
                    content = ""
            elif "<|message|>" in content_part:
                role = content_part.split("<|message|>")[0].strip()
                content = content_part.split("<|message|>")[1].strip()
            else:
                continue

            if not role:
                continue

            # Handle assistant analysis channel - store for potential merging with final
            if role == "assistant" and channel == "analysis":
                pending_analysis = content
                # Continue processing to see if final channel follows in this parse
                continue

            # Handle assistant final channel - merge with pending analysis if present
            if role == "assistant" and channel == "final":
                if pending_analysis:
                    content = f"<think>{pending_analysis}</think>{content}"
                    pending_analysis = None
                if content:
                    conversation.append(
                        NPSteerChatMessage(
                            role=role,
                            content=content,
                        )
                    )
                continue

            # Non-analysis, non-final messages
            if content:
                conversation.append(
                    NPSteerChatMessage(
                        role=role,
                        content=content,
                    )
                )

        # If there's pending analysis with no final yet, stream it as a think message
        # This enables real-time streaming of thinking content before final arrives
        if pending_analysis:
            conversation.append(
                NPSteerChatMessage(
                    role="assistant",
                    content=f"<think>{pending_analysis}</think>",
                )
            )

        return conversation

    # case: deepseek r1 distill llama 8b
    if custom_hf_model_id == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":
        for token in tokens:
            if current_content:
                if token == 128011:
                    if current_role:
                        conversation.append(
                            NPSteerChatMessage(
                                role=current_role,
                                content=tokenizer.decode(current_content).strip(),
                            )
                        )
                    current_content = []
                    current_role = "user"
                    continue
                if token == 128012:
                    if current_role:
                        conversation.append(
                            NPSteerChatMessage(
                                role=current_role,
                                content=tokenizer.decode(current_content).strip(),
                            )
                        )
                    current_content = []
                    current_role = "assistant"
                    continue
                if token == tokenizer.bos_token_id or token == tokenizer.eos_token_id:
                    continue
                current_content.append(token)
            # no current content, just append this token
            else:
                if token == 128011:
                    current_role = "user"
                elif token == 128012:
                    current_role = "assistant"
                elif (
                    token != tokenizer.bos_token_id and token != tokenizer.eos_token_id
                ):
                    current_content.append(token)
        # add the last content
        if current_content and current_role:
            conversation.append(
                NPSteerChatMessage(
                    role=current_role,
                    content=tokenizer.decode(current_content).strip(),
                )
            )
    
    # Llama 3.3 Instruct uses header tokens similar to Llama 3.1
    elif hasattr(tokenizer, "name_or_path") and "llama-3" in tokenizer.name_or_path.lower():
        # Llama 3.3 uses special tokens: 128006 (start_header), 128007 (end_header)
        # Format: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
        START_HEADER_ID = 128006
        END_HEADER_ID = 128007
        EOT_ID = 128009

        i = 0
        while i < len(tokens):
            token = tokens[i]

            # Look for start of a message (start_header_id)
            if token == START_HEADER_ID:
                # Extract role (tokens between start_header and end_header)
                role_tokens = []
                i += 1
                while i < len(tokens) and tokens[i] != END_HEADER_ID:
                    role_tokens.append(tokens[i])
                    i += 1

                if i < len(tokens) and tokens[i] == END_HEADER_ID:
                    i += 1  # Skip end_header_id

                    # Extract content (tokens until eot_id)
                    content_tokens = []
                    while i < len(tokens) and tokens[i] != EOT_ID:
                        content_tokens.append(tokens[i])
                        i += 1

                    if role_tokens and content_tokens:
                        role = tokenizer.decode(role_tokens).strip()
                        content = tokenizer.decode(content_tokens).strip()

                        # Strip Llama 3's auto-injected knowledge cutoff preamble from system messages
                        # to prevent duplication when apply_chat_template is called again
                        if role == "system":
                            content = _strip_llama3_system_preamble(content)

                        if role and content:
                            conversation.append(
                                NPSteerChatMessage(
                                    role=role,
                                    content=content,
                                )
                            )

                    if i < len(tokens) and tokens[i] == EOT_ID:
                        i += 1  # Skip eot_id
                    continue

            i += 1

    # no chat template, assume we are using the generic chat template to generate the conversation
    elif not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        # the chat template is format <|im_start|>{role}\n{content}<|im_end|>\n
        # Parse the text directly using string methods
        # Split by <|im_start|> to get conversation turns
        parts = text.split("<|im_start|>")

        for part in parts[1:]:  # Skip first empty part
            if not part.strip():
                continue

            # Find the end marker
            if "<|im_end|>" in part:
                content_part = part.split("<|im_end|>")[0]

                # Split by first newline to separate role from content
                if "\n" in content_part:
                    role, content = content_part.split("\n", 1)
                    role = role.strip()
                    content = content.strip()

                    if role and content:
                        conversation.append(
                            NPSteerChatMessage(
                                role=role,
                                content=content,
                            )
                        )

    # Llama 3.3 Instruct uses header tokens similar to Llama 3.1
    elif hasattr(tokenizer, "name_or_path") and "Llama-3" in tokenizer.name_or_path:
        # Llama 3.3 uses special tokens: 128006 (start_header), 128007 (end_header)
        # Format: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
        START_HEADER_ID = 128006
        END_HEADER_ID = 128007
        EOT_ID = 128009

        i = 0
        while i < len(tokens):
            token = tokens[i]

            # Look for start of a message (start_header_id)
            if token == START_HEADER_ID:
                # Extract role (tokens between start_header and end_header)
                role_tokens = []
                i += 1
                while i < len(tokens) and tokens[i] != END_HEADER_ID:
                    role_tokens.append(tokens[i])
                    i += 1

                if i < len(tokens) and tokens[i] == END_HEADER_ID:
                    i += 1  # Skip end_header_id

                    # Extract content (tokens until eot_id)
                    content_tokens = []
                    while i < len(tokens) and tokens[i] != EOT_ID:
                        content_tokens.append(tokens[i])
                        i += 1

                    if role_tokens and content_tokens:
                        role = tokenizer.decode(role_tokens).strip()
                        content = tokenizer.decode(content_tokens).strip()

                        if role and content:
                            conversation.append(
                                NPSteerChatMessage(
                                    role=role,
                                    content=content,
                                )
                            )

                    if i < len(tokens) and tokens[i] == EOT_ID:
                        i += 1  # Skip eot_id
                    continue

            i += 1

    # only other one right now is Gemma 2 Instruct (2B and 9B)
    else:
        # Get special token IDs directly from the tokenizer
        special_token_ids = config.steer_special_token_ids

        if special_token_ids is None:
            special_token_ids = set()

        for token in tokens:
            # first case is to check it's a special token that we append to the conversation
            if token in special_token_ids:
                if current_role and current_content:
                    item = NPSteerChatMessage(
                        role=current_role,
                        content=tokenizer.decode(current_content).strip(),
                    )
                    conversation.append(item)
                    current_content = []
                # no role or content yet, ignore the token
                current_role = None
            # second case is to check if it's a role token
            elif current_role is None:
                current_role = tokenizer.decode([token])
            # third case is to check if it's a content token
            else:
                current_content.append(token)

        # Add the last turn if exists
        if current_role and current_content:
            conversation.append(
                NPSteerChatMessage(
                    role=current_role,
                    content=tokenizer.decode(current_content).strip(),
                )
            )

    return conversation


def apply_generic_chat_template(
    messages: list[dict[str, str]], add_generation_prompt: bool = True
) -> str:
    """
    In case the model's tokenizer does not come with a chat template, we apply a generic chatML template.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        add_generation_prompt: Whether to add the assistant generation prompt

    Returns:
        str: Formatted chat string ready for tokenization
    """
    formatted_text = ""

    for message in messages:
        role = message["role"]
        content = message["content"]
        formatted_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    if add_generation_prompt:
        formatted_text += "<|im_start|>assistant\n"

    return formatted_text


class OrthogonalProjector:
    """Performs orthogonal projection steering for language model activations.

    This class implements low-rank orthogonal projection-based steering by projecting
    activations onto and orthogonal to a steering direction.

    Attributes:
        steering_vector: The direction to project onto/orthogonal to
        _P: Cached projection matrix
        _orthogonal_complement: Cached orthogonal complement matrix
    """

    def __init__(self, steering_vector: torch.Tensor):
        """Initializes projector with a steering vector.

        Args:
            steering_vector: Vector defining steering direction, shape (d,)
                           where d is activation dimension

        Raises:
            ValueError: If steering vector contains inf/nan values
        """
        self._P = None
        self._orthogonal_complement = None
        self.steering_vector = steering_vector.unsqueeze(1)

    def get_P(self) -> torch.Tensor:
        """Computes or returns cached projection matrix.

        Returns:
            Projection matrix P = vv^T/||v||^2, shape (d,d)

        Raises:
            ValueError: If projection computation fails or results in inf/nan
        """
        if self._P is None:
            # Compute the squared norm of the steering vector
            v_norm_squared = torch.sum(self.steering_vector * self.steering_vector)

            # Check for zero norm to avoid division by zero
            if v_norm_squared == 0:
                raise ValueError("Cannot create projection matrix from zero vector")

            # Compute the projection matrix: P = vv^T / ||v||^2
            self._P = (
                torch.matmul(self.steering_vector, self.steering_vector.T)
                / v_norm_squared
            )

            if not torch.isfinite(self._P).all():
                raise ValueError("Projection matrix contains inf or nan values")

        return self._P

    def get_orthogonal_complement(self) -> torch.Tensor:
        """Computes or returns cached orthogonal complement matrix.

        Returns:
            Matrix I-P where P is projection matrix, shape (d,d)

        Raises:
            ValueError: If computation fails
        """
        if self._orthogonal_complement is None:
            P = self.get_P()  # This may raise ValueError
            I = torch.eye(P.shape[0], dtype=P.dtype, device=P.device)  # noqa
            self._orthogonal_complement = I - P
            if not torch.isfinite(self._orthogonal_complement).all():
                raise ValueError(
                    "Orthogonal complement matrix contains inf or nan values"
                )
        return self._orthogonal_complement

    def project(
        self, activations: torch.Tensor, strength_multiplier: float = 1.0
    ) -> torch.Tensor:
        """Projects activations using orthogonal decomposition.

        Decomposes activations into components parallel and orthogonal to steering direction,
        then recombines with optional scaling of parallel component.

        Args:
            activations: Input activations to project, shape (d,)
            strength_multiplier: Scaling factor for parallel component

        Returns:
            Projected activations = (I-P)h + strength*Ph, shape (d,)
        """
        P = self.get_P()
        orthogonal_complement = self.get_orthogonal_complement()
        # use same dtype as activations
        orthogonal_complement = orthogonal_complement.to(activations.dtype)
        P = P.to(activations.dtype)
        return torch.matmul(
            activations, orthogonal_complement.T
        ) + strength_multiplier * torch.matmul(activations, P.T)
