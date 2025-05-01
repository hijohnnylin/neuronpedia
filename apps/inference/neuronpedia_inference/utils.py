import gc
import logging
import os

import torch
from psutil import Process

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
