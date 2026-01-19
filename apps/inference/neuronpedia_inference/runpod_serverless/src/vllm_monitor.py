"""
vLLM Health Monitoring Utility

Provides real-time stats about the vLLM engine including:
- Active/waiting requests
- GPU memory usage
- System RAM
- Active threads
- Request queue status

Usage:
    from vllm_monitor import VLLMMonitor

    # Create monitor from model manager
    monitor = VLLMMonitor(model_manager)

    # Get current stats
    stats = await monitor.get_stats()

    # Start background logging (logs every N seconds)
    monitor.start_background_logging(interval=5.0)
"""

import asyncio
import concurrent.futures
import logging
import os
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any

logger = logging.getLogger(__name__)

# Thread pool for running blocking operations (pynvml, psutil) without blocking the event loop
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="vllm-monitor")


@dataclass
class GPUStats:
    """GPU memory and utilization stats."""
    id: int
    name: str
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    gpu_utilization: float | None = None  # May not always be available


@dataclass
class VLLMStats:
    """Complete vLLM health stats."""
    timestamp: float
    
    # vLLM engine stats
    num_running_requests: int = 0
    num_waiting_requests: int = 0
    num_swapped_requests: int = 0
    gpu_cache_usage_percent: float = 0.0
    cpu_cache_usage_percent: float = 0.0
    
    # System stats
    system_ram_used_mb: float = 0.0
    system_ram_total_mb: float = 0.0
    system_ram_percent: float = 0.0
    process_ram_mb: float = 0.0
    
    # GPU stats
    gpu_stats: list[GPUStats] | None = None
    
    # Thread info
    active_threads: int = 0
    thread_names: list[str] | None = None
    
    # Asyncio info
    pending_tasks: int = 0
    
    # Engine state
    engine_initialized: bool = False
    model_name: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = asdict(self)
        if self.gpu_stats:
            d['gpu_stats'] = [asdict(g) for g in self.gpu_stats]
        return d
    
    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            f"=== vLLM Health Stats @ {time.strftime('%H:%M:%S', time.localtime(self.timestamp))} ===",
            f"Engine: {'READY' if self.engine_initialized else 'NOT INITIALIZED'} | Model: {self.model_name}",
            f"Requests: running={self.num_running_requests}, waiting={self.num_waiting_requests}, swapped={self.num_swapped_requests}",
            f"KV Cache: GPU={self.gpu_cache_usage_percent:.1f}%, CPU={self.cpu_cache_usage_percent:.1f}%",
            f"RAM: process={self.process_ram_mb:.0f}MB, system={self.system_ram_used_mb:.0f}/{self.system_ram_total_mb:.0f}MB ({self.system_ram_percent:.1f}%)",
        ]
        
        if self.gpu_stats:
            for g in self.gpu_stats:
                lines.append(
                    f"GPU[{g.id}] {g.name}: {g.memory_used_mb:.0f}/{g.memory_total_mb:.0f}MB ({g.memory_percent:.1f}%)"
                    + (f" util={g.gpu_utilization:.0f}%" if g.gpu_utilization is not None else "")
                )
        
        lines.append(f"Threads: {self.active_threads} | Async tasks: {self.pending_tasks}")
        
        return "\n".join(lines)


class VLLMMonitor:
    """
    Monitor for vLLM engine health and performance.
    
    Example:
        monitor = VLLMMonitor(model_manager)
        stats = await monitor.get_stats()
        print(stats.summary())
    """
    
    def __init__(self, model_manager=None):
        """
        Initialize the monitor.
        
        Args:
            model_manager: ModelManager instance with loaded VLLMSteerModel
        """
        self._model_manager = model_manager
        self._background_task: asyncio.Task | None = None
        self._stop_background = False
    
    def set_model_manager(self, model_manager):
        """Set or update the model manager."""
        self._model_manager = model_manager
    
    def _get_gpu_stats(self) -> list[GPUStats]:
        """Get GPU memory and utilization stats using nvidia-smi or pynvml."""
        gpu_stats = []
        
        # Try pynvml first (more reliable)
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Try to get utilization
                util = None
                try:
                    util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    util = util_info.gpu
                except Exception:
                    pass
                
                gpu_stats.append(GPUStats(
                    id=i,
                    name=name,
                    memory_used_mb=mem_info.used / (1024 * 1024),
                    memory_total_mb=mem_info.total / (1024 * 1024),
                    memory_percent=(mem_info.used / mem_info.total) * 100,
                    gpu_utilization=util,
                ))
            
            pynvml.nvmlShutdown()
            return gpu_stats
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"pynvml failed: {e}")
        
        # Fallback to torch.cuda
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    allocated = torch.cuda.memory_allocated(i)
                    total = props.total_memory
                    
                    gpu_stats.append(GPUStats(
                        id=i,
                        name=props.name,
                        memory_used_mb=allocated / (1024 * 1024),
                        memory_total_mb=total / (1024 * 1024),
                        memory_percent=(allocated / total) * 100 if total > 0 else 0,
                        gpu_utilization=None,
                    ))
        except Exception as e:
            logger.debug(f"torch.cuda stats failed: {e}")
        
        return gpu_stats
    
    def _get_system_stats(self) -> tuple[float, float, float, float]:
        """Get system RAM stats. Returns (used_mb, total_mb, percent, process_mb)."""
        try:
            import psutil
            
            mem = psutil.virtual_memory()
            process = psutil.Process(os.getpid())
            process_mem = process.memory_info().rss
            
            return (
                mem.used / (1024 * 1024),
                mem.total / (1024 * 1024),
                mem.percent,
                process_mem / (1024 * 1024),
            )
        except ImportError:
            return (0, 0, 0, 0)
        except Exception as e:
            logger.debug(f"psutil stats failed: {e}")
            return (0, 0, 0, 0)
    
    def _get_thread_info(self) -> tuple[int, list[str]]:
        """Get active thread count and names."""
        threads = threading.enumerate()
        return len(threads), [t.name for t in threads]
    
    def _get_async_task_count(self) -> int:
        """Get number of pending asyncio tasks."""
        try:
            loop = asyncio.get_running_loop()
            return len(asyncio.all_tasks(loop))
        except RuntimeError:
            return 0
    
    def _get_vllm_engine_stats_sync(self) -> dict[str, Any]:
        """Get vLLM engine internal stats (sync version for basic info only).
        
        Note: We intentionally don't call RPCs here to avoid interfering with
        active generation requests. Only basic model info is retrieved.
        """
        stats = {
            'initialized': False,
            'model_name': '',
            'num_running': 0,
            'num_waiting': 0,
            'num_swapped': 0,
            'gpu_cache_percent': 0.0,
            'cpu_cache_percent': 0.0,
        }
        
        if self._model_manager is None:
            return stats
        
        try:
            model = self._model_manager.model
            if model is None:
                return stats
            
            # Get the inner steerllm model
            inner = model._inner
            stats['model_name'] = inner._model_name
            
            # Check if engine is initialized
            engine = inner._engine
            if engine is None:
                return stats
            
            stats['initialized'] = True
            
            # Note: We don't call _collective_rpc here because:
            # 1. It could interfere with active generation requests
            # 2. The "get_health_stats" RPC handler doesn't exist in the worker
            # For detailed vLLM stats, use vLLM's built-in /metrics endpoint
            
        except Exception as e:
            logger.debug(f"Error getting vLLM engine stats: {e}")
        
        return stats
    
    def _collect_all_blocking_stats(self) -> tuple[list[GPUStats], tuple[float, float, float, float], tuple[int, list[str]], dict[str, Any]]:
        """Collect all blocking stats in one call (runs in thread pool)."""
        gpu_stats = self._get_gpu_stats()
        ram_stats = self._get_system_stats()
        thread_info = self._get_thread_info()
        engine_stats = self._get_vllm_engine_stats_sync()
        return gpu_stats, ram_stats, thread_info, engine_stats
    
    async def get_stats(self) -> VLLMStats:
        """
        Get current health stats.
        
        All blocking operations (pynvml, psutil) run in a thread pool to avoid
        blocking the event loop during streaming requests.
        
        Returns:
            VLLMStats with all available metrics
        """
        # Run blocking operations in thread pool to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        gpu_stats, ram_stats, thread_info, engine_stats = await loop.run_in_executor(
            _executor, self._collect_all_blocking_stats
        )
        
        ram_used, ram_total, ram_percent, process_ram = ram_stats
        thread_count, thread_names = thread_info
        pending_tasks = self._get_async_task_count()  # This is fast, no need for thread pool
        
        return VLLMStats(
            timestamp=time.time(),
            num_running_requests=engine_stats['num_running'],
            num_waiting_requests=engine_stats['num_waiting'],
            num_swapped_requests=engine_stats['num_swapped'],
            gpu_cache_usage_percent=engine_stats['gpu_cache_percent'],
            cpu_cache_usage_percent=engine_stats['cpu_cache_percent'],
            system_ram_used_mb=ram_used,
            system_ram_total_mb=ram_total,
            system_ram_percent=ram_percent,
            process_ram_mb=process_ram,
            gpu_stats=gpu_stats if gpu_stats else None,
            active_threads=thread_count,
            thread_names=thread_names,
            pending_tasks=pending_tasks,
            engine_initialized=engine_stats['initialized'],
            model_name=engine_stats['model_name'],
        )
    
    async def log_stats(self, level: int = logging.INFO):
        """Log current stats."""
        stats = await self.get_stats()
        logger.log(level, stats.summary())
    
    async def _background_logging_loop(self, interval: float):
        """Background loop that logs stats periodically."""
        while not self._stop_background:
            try:
                await self.log_stats()
            except Exception as e:
                logger.error(f"Error in background stats logging: {e}")
            
            await asyncio.sleep(interval)
    
    def start_background_logging(self, interval: float = 10.0):
        """
        Start background logging of stats.
        
        Args:
            interval: Seconds between log entries
        """
        if self._background_task is not None:
            logger.warning("Background logging already running")
            return
        
        self._stop_background = False
        
        try:
            loop = asyncio.get_running_loop()
            self._background_task = loop.create_task(
                self._background_logging_loop(interval)
            )
            logger.info(f"Started background vLLM stats logging (interval={interval}s)")
        except RuntimeError:
            logger.error("No running event loop - call start_background_logging from async context")
    
    def stop_background_logging(self):
        """Stop background logging."""
        self._stop_background = True
        if self._background_task is not None:
            self._background_task.cancel()
            self._background_task = None
            logger.info("Stopped background vLLM stats logging")


# Global monitor instance
_global_monitor: VLLMMonitor | None = None


def get_monitor() -> VLLMMonitor:
    """Get or create the global monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = VLLMMonitor()
    return _global_monitor


async def get_health_stats(model_manager=None) -> VLLMStats:
    """
    Convenience function to get health stats.
    
    Args:
        model_manager: Optional ModelManager to use
        
    Returns:
        VLLMStats with current health metrics
    """
    monitor = get_monitor()
    if model_manager is not None:
        monitor.set_model_manager(model_manager)
    return await monitor.get_stats()


async def log_health_stats(model_manager=None):
    """Convenience function to log health stats."""
    monitor = get_monitor()
    if model_manager is not None:
        monitor.set_model_manager(model_manager)
    await monitor.log_stats()

