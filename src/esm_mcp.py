"""
Model Context Protocol (MCP) for esm

This MCP server provides comprehensive protein structure analysis tools using ESM (Evolutionary Scale Modeling) models.
It combines structural dataset analysis and contact prediction capabilities for advanced protein research workflows.
The toolkit enables researchers to analyze protein structures, predict contacts, and visualize molecular interactions.

This MCP Server contains tools extracted from the following tutorial files:
1. esm_embeddings
    - esm_extract_embeddings_from_csv: Extract ESM embeddings from a CSV file containing protein sequences
"""

from fastmcp import FastMCP

# Import statements (alphabetical order)
from tools.esm_embeddings import esm_embeddings_mcp

# Server definition and mounting
mcp = FastMCP(name="esm")
mcp.mount(esm_embeddings_mcp)

if __name__ == "__main__":
    mcp.run()