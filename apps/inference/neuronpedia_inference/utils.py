import logging

import torch
from psutil import Process

logger = logging.getLogger(__name__)

process_for_logging_memory = Process()


# Sometimes CUDA just crashes and refuses to do anything ("CUDA assertion"), so we
# frequently call this to check if this is the case. If it has crashed, we force-kill the
# process to restart the server.
# There *has* to be a better way to do this?
def checkCudaError(device: str | None = None):
    if device is None:
        device = get_device()[0]

    if device == "cuda":
        try:
            # Every visible card: a sharded model puts weights and work on all of them, so
            # probing only cuda:0 would both under-report memory and miss a poisoned
            # context on another shard, which is the failure this is here to catch.
            used_mb = []
            for index in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(torch.device(f"cuda:{index}"))
                used_mb.append((total - free) / 1024**2)
            logger.info("Memory Used: %s", ", ".join(f"{used:.2f} MB" for used in used_mb))
        except RuntimeError as e:
            if "CUDA error" in str(e) or "CUDA assertion" in str(e):
                # Kill child procs (e.g. vLLM EngineCore) to reclaim GPU, then exit
                # so a supervisor restarts us. See resilience.terminate_for_restart.
                from neuronpedia_inference.resilience import terminate_for_restart

                terminate_for_restart(f"checkCudaError: {e}")
    elif device == "mps":
        logger.info(f"Memory Used: {torch.mps.current_allocated_memory() / (1024**2):.2f} MB")
    else:
        logger.info(f"Memory Used: {(process_for_logging_memory.memory_info().rss / (1024**2)):.2f} MB")


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
