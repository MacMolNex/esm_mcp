"""
Job representation for the queue system.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum
import uuid
import asyncio


class JobStatus(Enum):
    """Status of a job in the queue."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class Job:
    """Represents a job in the queue.

    Attributes:
        tool_name: Human-readable name of the tool
        tool_module: Python module path (e.g., "tools.esm_llh")
        tool_function: Function name to call (e.g., "_esm_calculate_llh_impl")
        kwargs: Arguments to pass to the tool function
        requires_gpu: Whether this job needs GPU resources
        job_id: Unique identifier for this job
        status: Current status of the job
        assigned_gpu: GPU device index assigned to this job
        result: Result from the tool function (set on completion)
        error: Error message (set on failure)
    """

    # Tool identification
    tool_name: str
    tool_module: str
    tool_function: str

    # Arguments to pass to the tool
    kwargs: Dict[str, Any]

    # Resource requirements
    requires_gpu: bool = True

    # Job metadata
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = field(default=JobStatus.PENDING)

    # Assigned resources (set by queue manager)
    assigned_gpu: Optional[int] = None

    # Result storage
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    # Async event for completion notification
    # Note: This is created lazily to avoid issues with multiprocessing
    _completion_event: Optional[asyncio.Event] = field(
        default=None, repr=False, compare=False
    )

    @property
    def completion_event(self) -> asyncio.Event:
        """Get or create the completion event."""
        if self._completion_event is None:
            self._completion_event = asyncio.Event()
        return self._completion_event

    def to_worker_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for sending to worker process."""
        return {
            "job_id": self.job_id,
            "tool_module": self.tool_module,
            "tool_function": self.tool_function,
            "kwargs": self.kwargs,
        }

    def mark_completed(self, result: Dict[str, Any]) -> None:
        """Mark the job as completed with result."""
        self.status = JobStatus.COMPLETED
        self.result = result
        if self._completion_event:
            self._completion_event.set()

    def mark_failed(self, error: str) -> None:
        """Mark the job as failed with error message."""
        self.status = JobStatus.FAILED
        self.error = error
        if self._completion_event:
            self._completion_event.set()

    def mark_timeout(self) -> None:
        """Mark the job as timed out."""
        self.status = JobStatus.TIMEOUT
        self.error = "Job execution timed out"
        if self._completion_event:
            self._completion_event.set()
