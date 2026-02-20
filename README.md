# ESM MCP Server

**Protein fitness prediction using Facebook's ESM models via Docker**

An MCP (Model Context Protocol) server for protein fitness prediction with 5 core tools:
- Extract embeddings (supervised learning)
- Train fitness models
- Predict fitness on new sequences
- Calculate sequence-based likelihood scores (zero-shot)
- Calculate structure-based likelihood scores (zero-shot)

## Quick Start with Docker

### Approach 1: Pull Pre-built Image from GitHub

The fastest way to get started. A pre-built Docker image is automatically published to GitHub Container Registry on every release.

```bash
# Pull the latest image
docker pull ghcr.io/macromnex/esm_mcp:latest

# Register with Claude Code (runs as current user to avoid permission issues)
claude mcp add esm -- docker run --gpus all -i --rm --user $(id -u):$(id -g) -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro ghcr.io/macromnex/esm_mcp:latest
```

**Requirements:**
- Docker with GPU support (`nvidia-docker` or Docker with NVIDIA runtime)
- Claude Code installed

That's it! The ESM MCP server is now available in Claude Code.

---

### Approach 2: Build Docker Image Locally

Build the image yourself and install it into Claude Code. Useful for customization or offline environments.

```bash
# Clone the repository
git clone https://github.com/MacromNex/esm_mcp.git
cd esm_mcp

# Build the Docker image
docker build -t esm_mcp:local .

# Register with Claude Code (runs as current user to avoid permission issues)
claude mcp add esm -- docker run --gpus all -i --rm --user $(id -u):$(id -g) -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro esm_mcp:local
```

**Requirements:**
- Docker with GPU support
- Claude Code installed
- Git (to clone the repository)

**About the Docker Flags:**
- `--user $(id -u):$(id -g)` — Runs the container as your current user, so output files are owned by you (not root)
- `-v /etc/passwd:/etc/passwd:ro` and `-v /etc/group:/etc/group:ro` — Provides user/group info to the container

---

## Verify Installation

After adding the MCP server, you can verify it's working:

```bash
# List registered MCP servers
claude mcp list

# You should see 'esm' in the output
```

In Claude Code, you can now use all 5 ESM tools:
- `esm_extract_embeddings_from_csv`
- `esm_calculate_llh`
- `esm_if_calculate_llh`
- `esm_train_fitness_model`
- `esm_predict_fitness`

---

## Next Steps

- **Detailed documentation**: See [detail.md](detail.md) for comprehensive guides on:
  - Available MCP tools and parameters
  - Local Python environment setup (alternative to Docker)
  - Example workflows and use cases
  - ESM model options and device selection

---

## Usage Examples

Once registered, you can use the ESM tools directly in Claude Code. Here are some common workflows:

### Example 1: Extract Embeddings

```
Can you extract ESM embeddings from my protein sequences at /path/to/data.csv using the esm_extract_embeddings_from_csv tool? Use the esm2_t33_650M_UR50D model and output to /path/to/embeddings/
```

### Example 2: Zero-Shot Fitness Prediction (Sequence-based)

```
I have protein variants in /path/to/variants.csv and the wild-type sequence in /path/to/wt.fasta. Can you calculate ESM likelihood scores using esm_calculate_llh to predict fitness without training data? Save results to /path/to/output.csv
```

### Example 3: Structure-Based Fitness Prediction

```
I have variants in /path/to/variants.csv, wild-type sequence in /path/to/wt.fasta, and structure in /path/to/structure.pdb. Can you calculate ESM-IF likelihood scores using esm_if_calculate_llh to predict fitness with structural information?
```

### Example 4: Train a Fitness Model

```
I have embeddings in /path/to/data/ directory with fitness values. Can you train an ESM fitness model using esm_train_fitness_model with SVR as the head model and 60 PCA components? Save to /path/to/models/
```

### Example 5: Predict with Trained Model

```
I have new sequences in /path/to/new_data.csv. Can you predict their fitness using esm_predict_fitness with the trained model from /path/to/models/final_model/?
```

---

## Supported ESM Models

All tools support the following backbone models:
- `esm2_t33_650M_UR50D` (default, 650M parameters)
- `esm2_t36_3B_UR50D` (3B parameters)
- `esm1v_t33_650M_UR90S_1` through `esm1v_t33_650M_UR90S_5`

## GPU Support

Both Docker approaches fully support:
- Multi-GPU systems (specify device via `cuda:0`, `cuda:1`, etc.)
- Single GPU setup
- CPU-only inference (use `cpu` device)

---

## Troubleshooting

**Docker not found?**
```bash
docker --version  # Install Docker if missing
```

**GPU not accessible?**
- Ensure NVIDIA Docker runtime is installed
- Check with `docker run --gpus all ubuntu nvidia-smi`

**Claude Code not found?**
```bash
# Install Claude Code
npm install -g @anthropic-ai/claude-code
```

---

## License

Same as ESM models (Meta AI Research)
