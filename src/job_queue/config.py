"""
Configuration for the job queue system.
"""

from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class QueueConfig:
    """Configuration for the job queue system."""

    # Number of worker processes (default=1, max depends on available GPUs)
    max_workers: int = 1

    # List of GPU device indices to use (e.g., [0, 1] for 2 GPUs)
    gpu_devices: List[int] = field(default_factory=lambda: [0])

    # Worker idle timeout in seconds (after which worker terminates to free GPU)
    worker_idle_timeout: float = 60.0

    # Job timeout in seconds (max time a single job can run)
    job_timeout: float = 3600.0

    # Maximum queue size (0 = unlimited)
    max_queue_size: int = 0

    @classmethod
    def from_env(cls) -> "QueueConfig":
        """Load configuration from environment variables.

        Environment variables:
            ESM_MAX_WORKERS: Number of parallel GPU jobs (default: 1)
            ESM_GPU_DEVICES: Comma-separated GPU indices (default: "0")
            ESM_WORKER_IDLE_TIMEOUT: Seconds before idle worker terminates (default: 60)
            ESM_JOB_TIMEOUT: Max job runtime in seconds (default: 3600)
            ESM_MAX_QUEUE_SIZE: Max queued jobs, 0=unlimited (default: 0)
        """
        gpu_devices_str = os.environ.get("ESM_GPU_DEVICES", "0")
        gpu_devices = [int(x.strip()) for x in gpu_devices_str.split(",") if x.strip()]

        return cls(
            max_workers=int(os.environ.get("ESM_MAX_WORKERS", "1")),
            gpu_devices=gpu_devices,
            worker_idle_timeout=float(os.environ.get("ESM_WORKER_IDLE_TIMEOUT", "60")),
            job_timeout=float(os.environ.get("ESM_JOB_TIMEOUT", "3600")),
            max_queue_size=int(os.environ.get("ESM_MAX_QUEUE_SIZE", "0")),
        )

    def validate(self) -> None:
        """Validate configuration values."""
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if not self.gpu_devices:
            raise ValueError("gpu_devices must contain at least one GPU index")
        if self.max_workers > len(self.gpu_devices):
            raise ValueError(
                f"max_workers ({self.max_workers}) cannot exceed "
                f"number of GPUs ({len(self.gpu_devices)})"
            )
        if self.worker_idle_timeout <= 0:
            raise ValueError("worker_idle_timeout must be positive")
        if self.job_timeout <= 0:
            raise ValueError("job_timeout must be positive")
