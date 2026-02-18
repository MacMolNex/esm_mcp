"""
Queue manager that orchestrates job queue, GPU assignment, and workers.
"""

import asyncio
import threading
import time
from queue import Queue, Empty, Full
from multiprocessing import Process, Queue as MPQueue
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .config import QueueConfig
from .job import Job, JobStatus
from .gpu_manager import GPUManager
from .worker import worker_process


@dataclass
class WorkerInfo:
    """Information about a worker process."""

    process: Process
    job_queue: MPQueue
    gpu_id: int
    worker_id: int
    last_job_time: float


class QueueManager:
    """Orchestrates job queue, GPU assignment, and worker management.

    Key responsibilities:
    1. Accept job submissions from MCP tools
    2. Queue jobs in FIFO order
    3. Spawn workers on-demand when jobs arrive
    4. Assign GPUs to workers
    5. Terminate idle workers to free GPU memory
    6. Return results to callers
    """

    def __init__(self, config: QueueConfig):
        """Initialize the queue manager.

        Args:
            config: Queue configuration
        """
        self.config = config
        config.validate()

        self.gpu_manager = GPUManager(config.gpu_devices)

        # Job tracking
        self._job_queue: Queue[Job] = Queue(
            maxsize=config.max_queue_size if config.max_queue_size > 0 else 0
        )
        self._pending_jobs: Dict[str, Job] = {}
        self._pending_jobs_lock = threading.Lock()

        # Worker tracking
        self._workers: Dict[int, WorkerInfo] = {}
        self._workers_lock = threading.Lock()
        self._result_queue: Optional[MPQueue] = None
        self._next_worker_id = 0

        # State management
        self._running = False
        self._started_lock = threading.Lock()

        # Background tasks
        self._dispatcher_task: Optional[asyncio.Task] = None
        self._result_collector_task: Optional[asyncio.Task] = None
        self._idle_checker_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the queue manager background tasks.

        This is called automatically on first job submission.
        """
        with self._started_lock:
            if self._running:
                return
            self._running = True

        # Create result queue for workers
        self._result_queue = MPQueue()

        # Start background tasks
        self._dispatcher_task = asyncio.create_task(self._dispatcher_loop())
        self._result_collector_task = asyncio.create_task(self._result_collector_loop())
        self._idle_checker_task = asyncio.create_task(self._idle_checker_loop())

        print("[ESM Queue Manager] Started", flush=True)

    async def stop(self) -> None:
        """Stop the queue manager and terminate all workers."""
        with self._started_lock:
            if not self._running:
                return
            self._running = False

        print("[ESM Queue Manager] Stopping...", flush=True)

        # Cancel background tasks
        for task in [
            self._dispatcher_task,
            self._result_collector_task,
            self._idle_checker_task,
        ]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Terminate all workers
        await self._terminate_all_workers()

        print("[ESM Queue Manager] Stopped", flush=True)

    async def submit(self, job: Job) -> Dict[str, Any]:
        """Submit a job and wait for result.

        This is the main entry point called by MCP tools.

        Args:
            job: Job to submit

        Returns:
            Result dictionary from the tool function
        """
        # Ensure manager is running
        if not self._running:
            await self.start()

        # Register job for tracking
        with self._pending_jobs_lock:
            self._pending_jobs[job.job_id] = job

        # Add to queue (may block if queue is full)
        try:
            self._job_queue.put(job, timeout=10.0)
        except Full:
            job.mark_failed("Queue is full, try again later")
            with self._pending_jobs_lock:
                del self._pending_jobs[job.job_id]
            return {"status": "error", "error_message": job.error}

        print(f"[ESM Queue Manager] Job {job.job_id} queued: {job.tool_name}", flush=True)

        # Wait for completion with timeout
        try:
            await asyncio.wait_for(
                job.completion_event.wait(),
                timeout=self.config.job_timeout,
            )
        except asyncio.TimeoutError:
            job.mark_timeout()
            # Try to terminate the worker running this job
            await self._handle_job_timeout(job)

        # Return result
        if job.status == JobStatus.COMPLETED:
            return job.result
        else:
            return {
                "status": "error",
                "error_message": job.error or "Unknown error",
            }

    async def _dispatcher_loop(self) -> None:
        """Dispatch jobs from queue to available workers."""
        while self._running:
            try:
                # Try to get a job from the queue (non-blocking)
                try:
                    job = self._job_queue.get_nowait()
                except Empty:
                    await asyncio.sleep(0.1)
                    continue

                # Try to acquire a GPU
                gpu_id = self.gpu_manager.acquire_gpu(job.job_id)

                if gpu_id is None:
                    # No GPU available, put job back and wait
                    self._job_queue.put(job)
                    await asyncio.sleep(0.2)
                    continue

                # Assign GPU to job
                job.assigned_gpu = gpu_id
                job.status = JobStatus.RUNNING

                # Dispatch to worker
                await self._dispatch_to_worker(job)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[ESM Queue Manager] Dispatcher error: {e}", flush=True)
                await asyncio.sleep(1.0)

    async def _dispatch_to_worker(self, job: Job) -> None:
        """Dispatch a job to a worker process."""
        gpu_id = job.assigned_gpu

        # Check for existing worker on this GPU
        worker = None
        with self._workers_lock:
            for w in self._workers.values():
                if w.gpu_id == gpu_id and w.process.is_alive():
                    worker = w
                    break

        # Spawn new worker if needed
        if worker is None:
            worker = self._spawn_worker(gpu_id)

        # Send job to worker
        worker.job_queue.put(job.to_worker_dict())
        worker.last_job_time = time.time()

        print(
            f"[ESM Queue Manager] Job {job.job_id} dispatched to worker "
            f"{worker.worker_id} on GPU {gpu_id}",
            flush=True,
        )

    def _spawn_worker(self, gpu_id: int) -> WorkerInfo:
        """Spawn a new worker process for the given GPU."""
        with self._workers_lock:
            worker_id = self._next_worker_id
            self._next_worker_id += 1

        job_queue = MPQueue()

        process = Process(
            target=worker_process,
            args=(
                job_queue,
                self._result_queue,
                gpu_id,
                worker_id,
                self.config.worker_idle_timeout,
            ),
            daemon=True,
        )
        process.start()

        worker = WorkerInfo(
            process=process,
            job_queue=job_queue,
            gpu_id=gpu_id,
            worker_id=worker_id,
            last_job_time=time.time(),
        )

        with self._workers_lock:
            self._workers[worker_id] = worker

        print(
            f"[ESM Queue Manager] Spawned worker {worker_id} on GPU {gpu_id}",
            flush=True,
        )

        return worker

    async def _result_collector_loop(self) -> None:
        """Collect results from worker processes."""
        while self._running:
            try:
                # Check for results (non-blocking)
                try:
                    result = self._result_queue.get_nowait()
                except Empty:
                    await asyncio.sleep(0.05)
                    continue

                job_id = result["job_id"]

                with self._pending_jobs_lock:
                    job = self._pending_jobs.get(job_id)

                if job:
                    if result["status"] == "completed":
                        job.mark_completed(result["result"])
                        print(
                            f"[ESM Queue Manager] Job {job_id} completed",
                            flush=True,
                        )
                    else:
                        error = result.get("error", "Unknown error")
                        job.mark_failed(error)
                        print(
                            f"[ESM Queue Manager] Job {job_id} failed: {error}",
                            flush=True,
                        )

                    # Release GPU
                    if job.assigned_gpu is not None:
                        self.gpu_manager.release_gpu(job.assigned_gpu)

                    # Remove from pending
                    with self._pending_jobs_lock:
                        if job_id in self._pending_jobs:
                            del self._pending_jobs[job_id]

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[ESM Queue Manager] Result collector error: {e}", flush=True)
                await asyncio.sleep(0.1)

    async def _idle_checker_loop(self) -> None:
        """Check for and terminate idle workers."""
        while self._running:
            try:
                await asyncio.sleep(5.0)  # Check every 5 seconds

                current_time = time.time()
                workers_to_remove: List[int] = []

                with self._workers_lock:
                    for worker_id, worker in self._workers.items():
                        # Check if worker process has died
                        if not worker.process.is_alive():
                            workers_to_remove.append(worker_id)
                            continue

                        # Check if worker is idle and all jobs complete
                        if (
                            self._job_queue.empty()
                            and self.gpu_manager.get_gpu_for_job(
                                worker.job_queue.empty() and str(worker_id)
                            )
                            is None
                        ):
                            idle_time = current_time - worker.last_job_time
                            if idle_time > self.config.worker_idle_timeout:
                                # Send shutdown signal
                                try:
                                    worker.job_queue.put(None, timeout=1.0)
                                except Full:
                                    pass

                                worker.process.join(timeout=5.0)
                                if worker.process.is_alive():
                                    worker.process.terminate()
                                    worker.process.join(timeout=2.0)

                                workers_to_remove.append(worker_id)
                                print(
                                    f"[ESM Queue Manager] Terminated idle worker "
                                    f"{worker_id} on GPU {worker.gpu_id}",
                                    flush=True,
                                )

                    # Remove terminated workers
                    for worker_id in workers_to_remove:
                        if worker_id in self._workers:
                            del self._workers[worker_id]

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[ESM Queue Manager] Idle checker error: {e}", flush=True)

    async def _handle_job_timeout(self, job: Job) -> None:
        """Handle a job that has timed out."""
        print(f"[ESM Queue Manager] Job {job.job_id} timed out", flush=True)

        # Release GPU
        if job.assigned_gpu is not None:
            self.gpu_manager.release_gpu(job.assigned_gpu)

        # Try to find and terminate the worker
        with self._workers_lock:
            for worker_id, worker in list(self._workers.items()):
                if worker.gpu_id == job.assigned_gpu:
                    worker.process.terminate()
                    worker.process.join(timeout=2.0)
                    del self._workers[worker_id]
                    print(
                        f"[ESM Queue Manager] Terminated worker {worker_id} "
                        f"due to job timeout",
                        flush=True,
                    )
                    break

    async def _terminate_all_workers(self) -> None:
        """Terminate all worker processes."""
        with self._workers_lock:
            for worker in self._workers.values():
                # Send shutdown signal
                try:
                    worker.job_queue.put(None, timeout=1.0)
                except Full:
                    pass

                worker.process.join(timeout=5.0)
                if worker.process.is_alive():
                    worker.process.terminate()
                    worker.process.join(timeout=2.0)

            self._workers.clear()

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current status of the queue system."""
        with self._workers_lock:
            active_workers = sum(
                1 for w in self._workers.values() if w.process.is_alive()
            )

        with self._pending_jobs_lock:
            pending_count = len(self._pending_jobs)

        return {
            "running": self._running,
            "queued_jobs": self._job_queue.qsize(),
            "pending_jobs": pending_count,
            "active_workers": active_workers,
            "available_gpus": self.gpu_manager.get_available_count(),
            "gpus_in_use": self.gpu_manager.get_in_use_count(),
        }
