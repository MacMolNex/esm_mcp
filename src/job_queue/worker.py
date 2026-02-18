"""
Worker process for executing GPU jobs.

Each worker runs in a separate process with CUDA_VISIBLE_DEVICES set
to a single GPU. When the worker terminates, all GPU memory is released.
"""

import os
import sys
import gc
import importlib
import traceback
from multiprocessing import Queue as MPQueue
from typing import Any, Dict, Optional


def worker_process(
    job_queue: MPQueue,
    result_queue: MPQueue,
    gpu_device: int,
    worker_id: int,
    idle_timeout: float = 60.0,
):
    """Worker process that executes jobs on assigned GPU.

    This process runs in isolation to ensure GPU memory is fully
    released when the process terminates.

    Args:
        job_queue: Queue to receive jobs from
        result_queue: Queue to send results to
        gpu_device: GPU device index to use
        worker_id: Unique identifier for this worker
        idle_timeout: Seconds to wait for new job before terminating
    """
    # Set GPU visibility BEFORE any torch imports
    # This must happen at the very start of the process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)

    # Now import torch - it will only see the assigned GPU as cuda:0
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Log worker startup
    _log(f"Worker {worker_id} started on GPU {gpu_device} (visible as {device})")

    while True:
        try:
            # Get job from queue (blocking with timeout)
            try:
                job_data = job_queue.get(timeout=idle_timeout)
            except Exception:
                # Queue.Empty or other timeout - terminate worker
                _log(f"Worker {worker_id} idle timeout, terminating")
                break

            if job_data is None:
                # Shutdown signal received
                _log(f"Worker {worker_id} received shutdown signal")
                break

            job_id = job_data["job_id"]
            tool_module = job_data["tool_module"]
            tool_function = job_data["tool_function"]
            kwargs = job_data["kwargs"]

            _log(f"Worker {worker_id} executing job {job_id}: {tool_function}")

            try:
                # Import and execute the tool function
                module = importlib.import_module(tool_module)
                func = getattr(module, tool_function)

                # Inject device parameter - always cuda:0 since we only see one GPU
                kwargs["device"] = device

                # Execute the tool function
                result = func(**kwargs)

                # Clear GPU cache after job
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                # Send success result
                result_queue.put(
                    {
                        "job_id": job_id,
                        "status": "completed",
                        "result": result,
                    }
                )
                _log(f"Worker {worker_id} completed job {job_id}")

            except Exception as e:
                # Job failed - send error result
                error_msg = f"{type(e).__name__}: {str(e)}"
                error_traceback = traceback.format_exc()
                _log(f"Worker {worker_id} job {job_id} failed: {error_msg}")

                result_queue.put(
                    {
                        "job_id": job_id,
                        "status": "failed",
                        "error": error_msg,
                        "traceback": error_traceback,
                    }
                )

                # Clear GPU cache even on failure
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            # Unexpected error in worker loop
            _log(f"Worker {worker_id} unexpected error: {e}")
            continue

    # Final cleanup before process exits
    _log(f"Worker {worker_id} cleaning up")
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    _log(f"Worker {worker_id} exiting")


def _log(message: str) -> None:
    """Simple logging function for worker process."""
    # Use print with flush for immediate output
    print(f"[ESM Queue Worker] {message}", flush=True)
