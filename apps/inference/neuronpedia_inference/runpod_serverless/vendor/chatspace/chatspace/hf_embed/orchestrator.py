"""Process orchestration, lifecycle management, and signal handling."""

from __future__ import annotations

import logging
import multiprocessing as mp
import signal
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

from tqdm import tqdm


class ProcessStatus(Enum):
    """Status of a managed process."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class ManagedProcess:
    """Metadata for a managed process."""

    name: str
    process: mp.Process
    status: ProcessStatus = ProcessStatus.PENDING


@dataclass
class ProgressUpdate:
    """Progress update message from worker processes."""

    rows_processed: int
    stage: str = "encoder"


class ProcessController:
    """Manages process lifecycle, signal handling, and progress tracking."""

    def __init__(self) -> None:
        self._processes: list[ManagedProcess] = []
        self._shutdown_event: Optional[mp.Event] = None
        self._original_sigint_handler: Optional[Any] = None
        self._original_sigterm_handler: Optional[Any] = None
        self._progress_bar: Optional[tqdm] = None
        self._shutdown_requested = False

    def register_process(self, name: str, process: mp.Process) -> None:
        """Register a process for lifecycle management."""
        self._processes.append(ManagedProcess(name=name, process=process))

    def set_shutdown_event(self, event: mp.Event) -> None:
        """Set the shared shutdown event for signaling workers."""
        self._shutdown_event = event

    def setup_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown."""
        self._original_sigint_handler = signal.signal(signal.SIGINT, self._handle_signal)
        self._original_sigterm_handler = signal.signal(signal.SIGTERM, self._handle_signal)

    def restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        if self._original_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._original_sigint_handler)
        if self._original_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm_handler)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals (SIGINT/SIGTERM)."""
        if self._shutdown_requested:
            logging.warning("Shutdown already in progress, forcing termination...")
            self._force_terminate_all()
            raise KeyboardInterrupt("Forced shutdown")

        sig_name = "SIGINT" if signum == signal.SIGINT else f"signal {signum}"
        logging.info("Received %s, initiating graceful shutdown...", sig_name)
        self._shutdown_requested = True

        if self._shutdown_event is not None:
            self._shutdown_event.set()

    def start_all(self) -> None:
        """Start all registered processes."""
        for managed in self._processes:
            managed.process.start()
            managed.status = ProcessStatus.RUNNING
            logging.debug("Started process: %s (PID %d)", managed.name, managed.process.pid)

    def join_all(self, timeout: float = 30.0) -> None:
        """Join all processes with timeout, then terminate stragglers."""
        deadline = time.time() + timeout
        for managed in self._processes:
            if managed.status != ProcessStatus.RUNNING:
                continue

            remaining = max(0.0, deadline - time.time())
            managed.process.join(timeout=remaining)

            if managed.process.is_alive():
                logging.warning("Process %s did not exit within timeout, terminating...", managed.name)
                managed.process.terminate()
                managed.process.join(timeout=2.0)
                managed.status = ProcessStatus.TERMINATED
            else:
                if managed.process.exitcode == 0:
                    managed.status = ProcessStatus.COMPLETED
                else:
                    managed.status = ProcessStatus.FAILED
                    logging.warning(
                        "Process %s exited with code %s", managed.name, managed.process.exitcode
                    )

    def _force_terminate_all(self) -> None:
        """Forcefully terminate all processes."""
        for managed in self._processes:
            if managed.process.is_alive():
                logging.warning("Force terminating process: %s", managed.name)
                managed.process.terminate()
                managed.process.join(timeout=1.0)
                if managed.process.is_alive():
                    managed.process.kill()

    def check_for_failures(self) -> None:
        """Check if any process has failed and raise an exception."""
        for managed in self._processes:
            if not managed.process.is_alive() and managed.process.exitcode != 0:
                raise RuntimeError(
                    f"Process {managed.name} failed with exit code {managed.process.exitcode}"
                )

    def create_progress_bar(self, disable: bool = False, **kwargs: Any) -> tqdm:
        """Create and store a progress bar."""
        self._progress_bar = tqdm(disable=disable, **kwargs)
        return self._progress_bar

    def update_progress(self, n: int = 1) -> None:
        """Update progress bar if it exists."""
        if self._progress_bar is not None:
            self._progress_bar.update(n)

    def close_progress_bar(self) -> None:
        """Close the progress bar if it exists."""
        if self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None

    def monitor_progress_queue(
        self, progress_queue: mp.Queue[Any], stop_token: Any, timeout: float = 0.1
    ) -> None:
        """Monitor progress queue and update progress bar (non-blocking poll)."""
        try:
            while True:
                try:
                    msg = progress_queue.get(timeout=timeout)
                    if msg is stop_token:
                        break
                    if isinstance(msg, ProgressUpdate):
                        self.update_progress(msg.rows_processed)
                except Exception:
                    # Check if processes are still alive
                    if not any(p.process.is_alive() for p in self._processes):
                        break
                    continue
        except KeyboardInterrupt:
            pass

    def run_with_monitoring(
        self,
        main_work: Callable[[], Any],
        progress_queue: Optional[mp.Queue[Any]] = None,
        stop_token: Any = None,
    ) -> Any:
        """Run main work while monitoring process health and progress."""
        self.setup_signal_handlers()
        try:
            self.start_all()

            # If progress queue provided, monitor in background
            if progress_queue is not None and stop_token is not None:
                import threading

                monitor_thread = threading.Thread(
                    target=self.monitor_progress_queue,
                    args=(progress_queue, stop_token),
                    daemon=True,
                )
                monitor_thread.start()

            # Run main work
            result = main_work()

            return result

        except KeyboardInterrupt:
            logging.info("Keyboard interrupt detected, shutting down...")
            raise
        finally:
            self.restore_signal_handlers()
            self.close_progress_bar()
            self.join_all()
            self.check_for_failures()