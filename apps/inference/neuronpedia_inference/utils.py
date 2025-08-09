import gc
import logging
import os

import torch
from neuronpedia_inference_client.models.np_logprob import NPLogprob
from neuronpedia_inference_client.models.np_logprob_top import NPLogprobTop
from psutil import Process
from transformer_lens import HookedTransformer

logger = logging.getLogger(__name__)

process_for_logging_memory = Process()


# Sometimes CUDA just crashes and refuses to do anything ("CUDA assertion"), so we frequently call this to check if this is the case.
# If it has crashed, we force-kill the process to restart the server.
# There *has* to be a better way to do this?
def checkCudaError(device: str | None = None):
    if device is None:
        device = get_device()[0]

    if device == "cuda":
        try:
            free, total = torch.cuda.mem_get_info(torch.device("cuda:0"))
            mem_used_MB = (total - free) / 1024**2
            logger.info(f"Memory Used: {mem_used_MB:.2f} MB")
        except RuntimeError as e:
            if "CUDA error" in str(e) or "CUDA assertion" in str(e):
                logger.error(f"EXITING - CUDA error: {e}")
                torch.cuda.reset_peak_memory_stats()
                gc.collect()
                os._exit(1)
    elif device == "mps":
        logger.info(
            f"Memory Used: {torch.mps.current_allocated_memory() / (1024**2):.2f} MB"
        )
    else:
        logger.info(
            f"Memory Used: {(process_for_logging_memory.memory_info().rss / (1024**2)):.2f} MB"
        )


def get_device():
    device = "cpu"
    device_count = 1
    if torch.backends.mps.is_available():
        device = "mps"
    if torch.cuda.is_available():
        logger.info("cuda is available")
        device = "cuda"
        device_count = torch.cuda.device_count()

    return device, device_count


def make_logprob_from_logits(
    result: torch.Tensor,
    logits: torch.Tensor,
    model: HookedTransformer,
    n_logprobs: int = 10,
) -> NPLogprob:
    # Note: logits from generate_stream with return_logits=True has shape [batch_size, 1, vocab_size]
    # where the middle dimension is always 1 (only last position logits)
    log_probs = torch.log_softmax(logits, dim=-1)

    # Get the token that was generated
    token_id = result[0][-1].item()
    token_str = model.tokenizer.decode([token_id])  # type: ignore

    logprob = log_probs[0, 0, token_id].item()  # type: ignore

    # Get top k logprobs from the same position
    position_logprobs = log_probs[0, 0, :]
    top_k_logprobs, top_indices = torch.topk(position_logprobs, k=n_logprobs, dim=-1)

    top_logprobs = []
    for k in range(min(n_logprobs, len(top_indices))):
        current_token_id = top_indices[k].item()
        current_token_str = model.tokenizer.decode([current_token_id])  # type: ignore
        logprob_val = top_k_logprobs[k].item()

        # Replace NaN values with very small log probability
        if not torch.isfinite(torch.tensor(logprob_val)):
            logprob_val = -100.0

        top_logprobs.append(NPLogprobTop(token=current_token_str, logprob=logprob_val))

    return NPLogprob(
        token=token_str,
        logprob=logprob,
        top_logprobs=top_logprobs,
    )
