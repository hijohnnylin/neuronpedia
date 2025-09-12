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

    # handle llama3.1-8b-instruct
    elif hasattr(tokenizer, 'name_or_path') and tokenizer.name_or_path == "meta-llama/Llama-3.1-8B-Instruct":
        # Handle Llama 3.1 format with <|start_header_id|> and <|end_header_id|>
        # Token IDs: 128006 = <|start_header_id|>, 128007 = <|end_header_id|>
        in_header = False
        header_tokens = []
        
        for token in tokens:
            if token == tokenizer.bos_token_id:
                continue
            if token == tokenizer.eos_token_id:
                continue
            if token == 128006:  # <|start_header_id|>
                in_header = True
                header_tokens = []
                # Add any accumulated content before starting new role
                if current_role and current_content:
                    conversation.append(
                        NPSteerChatMessage(
                            role=current_role,
                            content=tokenizer.decode(current_content).strip(),
                        )
                    )
                    current_content = []
                continue
            elif token == 128007:  # <|end_header_id|>
                if in_header and header_tokens:
                    current_role = tokenizer.decode(header_tokens).strip()
                in_header = False
                header_tokens = []
                continue
            elif in_header:
                header_tokens.append(token)
            elif token == 128011:  # Legacy user token
                current_role = "user"
            elif token == 128012:  # Legacy assistant token
                current_role = "assistant"
            else:
                current_content.append(token)
        
        # Add final message if exists
        if current_role and current_content:
            conversation.append(
                NPSteerChatMessage(
                    role=current_role,
                    content=tokenizer.decode(current_content).strip(),
                )
            )

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
            self.steering_vector = torch.matmul(
                self.steering_vector, self.steering_vector.T
            )
            self._P = self.steering_vector

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
