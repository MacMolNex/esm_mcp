"""
GPU resource management for the job queue.
"""

import threading
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class GPUStatus:
    """Status of a single GPU."""

    device_id: int
    in_use: bool = False
    current_job_id: Optional[str] = None


class GPUManager:
    """Manages GPU allocation and deallocation.

    Thread-safe manager for tracking which GPUs are available and
    assigning them to jobs.
    """

    def __init__(self, gpu_devices: List[int]):
        """Initialize GPU manager with list of available GPU device indices.

        Args:
            gpu_devices: List of GPU indices, e.g., [0, 1] for cuda:0, cuda:1
        """
        self._lock = threading.Lock()
        self._gpus: Dict[int, GPUStatus] = {
            idx: GPUStatus(device_id=idx) for idx in gpu_devices
        }

    def acquire_gpu(self, job_id: str) -> Optional[int]:
        """Acquire an available GPU for a job.

        Args:
            job_id: ID of the job requesting a GPU

        Returns:
            GPU device index if available, None otherwise
        """
        with self._lock:
            for gpu_id, status in self._gpus.items():
                if not status.in_use:
                    status.in_use = True
                    status.current_job_id = job_id
                    return gpu_id
            return None

    def release_gpu(self, gpu_id: int) -> None:
        """Release a GPU back to the pool.

        Args:
            gpu_id: GPU device index to release
        """
        with self._lock:
            if gpu_id in self._gpus:
                self._gpus[gpu_id].in_use = False
                self._gpus[gpu_id].current_job_id = None

    def get_available_count(self) -> int:
        """Get number of available GPUs."""
        with self._lock:
            return sum(1 for s in self._gpus.values() if not s.in_use)

    def get_in_use_count(self) -> int:
        """Get number of GPUs currently in use."""
        with self._lock:
            return sum(1 for s in self._gpus.values() if s.in_use)

    def all_idle(self) -> bool:
        """Check if all GPUs are idle (not in use)."""
        with self._lock:
            return all(not s.in_use for s in self._gpus.values())

    def get_status(self) -> Dict[int, GPUStatus]:
        """Get a copy of current GPU status."""
        with self._lock:
            return {
                gpu_id: GPUStatus(
                    device_id=status.device_id,
                    in_use=status.in_use,
                    current_job_id=status.current_job_id,
                )
                for gpu_id, status in self._gpus.items()
            }

    def get_gpu_for_job(self, job_id: str) -> Optional[int]:
        """Get the GPU assigned to a specific job.

        Args:
            job_id: ID of the job

        Returns:
            GPU device index if found, None otherwise
        """
        with self._lock:
            for gpu_id, status in self._gpus.items():
                if status.current_job_id == job_id:
                    return gpu_id
            return None
