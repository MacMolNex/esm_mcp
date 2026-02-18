"""
Model Context Protocol (MCP) for esm

This MCP server provides comprehensive protein structure analysis tools using ESM (Evolutionary Scale Modeling) models.
It combines structural dataset analysis and contact prediction capabilities for advanced protein research workflows.
The toolkit enables researchers to analyze protein structures, predict contacts, and visualize molecular interactions.

This MCP Server contains tools extracted from the following tutorial files:
1. esm_embeddings
    - esm_extract_embeddings_from_csv: Extract ESM embeddings from a CSV file containing protein sequences
2. esm_llh
    - esm_calculate_llh: Calculate ESM log-likelihood for protein mutations
3. esm_if_llh
    - esm_if_calculate_llh: Calculate ESM-IF log-likelihood using protein structure
4. esm_train_fitness
    - esm_train_fitness_model: Train regression models on ESM embeddings for fitness prediction
5. esm_predict_fitness
    - esm_predict_fitness: Predict fitness values using pre-trained ESM-based models

Job Queue System:
    GPU-intensive tools (esm_embeddings, esm_llh, esm_if_llh) are managed by a job queue
    that ensures FIFO processing and automatic GPU assignment. Configure with environment
    variables:
        ESM_MAX_WORKERS: Number of parallel GPU jobs (default: 1)
        ESM_GPU_DEVICES: Comma-separated GPU indices (default: "0")
        ESM_WORKER_IDLE_TIMEOUT: Seconds before idle worker terminates (default: 60)
"""

from fastmcp import FastMCP
import multiprocessing

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Import queue system
from job_queue import QueueConfig, QueueManager

# Initialize queue manager with configuration from environment
config = QueueConfig.from_env()
queue_manager = QueueManager(config)

# Import GPU tools with queue wrappers
from tools.esm_embeddings import create_esm_embeddings_mcp
from tools.esm_if_llh import create_esm_if_llh_mcp
from tools.esm_llh import create_esm_llh_mcp

# Import CPU tools (no queue needed)
from tools.esm_predict_fitness import esm_predict_fitness_mcp
from tools.esm_train_fitness import esm_train_fitness_mcp

# Create queue-wrapped MCP instances for GPU tools
esm_embeddings_mcp = create_esm_embeddings_mcp(queue_manager)
esm_llh_mcp = create_esm_llh_mcp(queue_manager)
esm_if_llh_mcp = create_esm_if_llh_mcp(queue_manager)

# Server definition and mounting
mcp = FastMCP(name="esm_mcp")
mcp.mount(esm_embeddings_mcp)
mcp.mount(esm_llh_mcp)
mcp.mount(esm_if_llh_mcp)
mcp.mount(esm_train_fitness_mcp)
mcp.mount(esm_predict_fitness_mcp)

if __name__ == "__main__":
    print(f"[ESM MCP Server] Starting with config: max_workers={config.max_workers}, "
          f"gpu_devices={config.gpu_devices}, worker_idle_timeout={config.worker_idle_timeout}s")
    mcp.run()
