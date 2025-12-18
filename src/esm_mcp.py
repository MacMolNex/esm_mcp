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
"""

from fastmcp import FastMCP
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# Import statements (alphabetical order)
from tools.esm_embeddings import esm_embeddings_mcp
from tools.esm_if_llh import esm_if_llh_mcp
from tools.esm_llh import esm_llh_mcp
from tools.esm_predict_fitness import esm_predict_fitness_mcp
from tools.esm_train_fitness import esm_train_fitness_mcp

# Server definition and mounting
mcp = FastMCP(name="esm_mcp")
mcp.mount(esm_embeddings_mcp)
mcp.mount(esm_llh_mcp)
mcp.mount(esm_if_llh_mcp)
mcp.mount(esm_train_fitness_mcp)
mcp.mount(esm_predict_fitness_mcp)

if __name__ == "__main__":
    mcp.run()