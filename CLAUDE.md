# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ESM MCP Server — a Model Context Protocol (MCP) server for protein fitness prediction using Facebook's ESM (Evolutionary Scale Modeling) models. Provides both zero-shot (LLH-based) and supervised (embedding + regression) fitness prediction workflows.

## Setup & Running

```bash
# Quick setup (handles conda env, ESM repo clone, model downloads)
bash quick_setup.sh

# Activate environment
conda activate esm_mcp

# Run the MCP server
python src/server.py

# Run individual tools as CLI scripts (in scripts/ directory)
python scripts/esm_embeddings.py --help
```

**Docker:**
```bash
docker build -t esm_mcp .
```

There is no test suite or linter configured for this project.

## Architecture

### Server Entry Point: `src/server.py`
Uses **FastMCP** framework. Mounts 5 tool sub-servers onto a single `esm_mcp` MCP server. The multiprocessing start method is forced to `spawn` (required for CUDA in subprocesses).

### Two Categories of Tools

**GPU tools** (queue-managed via `src/job_queue/`):
- `esm_extract_embeddings_from_csv` — Extract ESM2/ESM1v embeddings from protein sequences
- `esm_calculate_llh` — Zero-shot fitness prediction via mutation log-likelihoods
- `esm_if_calculate_llh` — Structure-aware zero-shot prediction using ESM-IF and PDB files

**CPU tools** (run directly, no queue):
- `esm_train_fitness_model` — Train regression heads (SVR, RF, XGBoost, etc.) on embeddings
- `esm_predict_fitness` — Inference with trained models

Each GPU tool module exports a `create_*_mcp(queue_manager)` factory that returns a FastMCP sub-server with queue-wrapped endpoints. CPU tools export pre-built `*_mcp` FastMCP instances directly.

### Job Queue System (`src/job_queue/`)

Handles GPU resource management for concurrent MCP requests:
- **QueueManager**: FIFO dispatcher + result collector + idle checker (all async tasks)
- **GPUManager**: Thread-safe GPU allocation/release
- **Worker**: Isolated subprocess per GPU — sets `CUDA_VISIBLE_DEVICES` before torch import, calls `torch.cuda.empty_cache()` + `gc.collect()` after each job
- **Job**: Command object with async completion event for caller notification

Flow: MCP call → Job created → queued → dispatcher acquires GPU → dispatches to worker subprocess → result collected → GPU released → async event signals caller.

Configuration via environment variables:
```
ESM_MAX_WORKERS=1              # Parallel GPU workers
ESM_GPU_DEVICES=0              # Comma-separated GPU indices
ESM_WORKER_IDLE_TIMEOUT=60     # Seconds before idle worker exits
ESM_JOB_TIMEOUT=3600           # Max job runtime
ESM_MAX_QUEUE_SIZE=0           # 0 = unlimited
```

### CLI Scripts (`scripts/`)

Mirror the MCP tools as standalone CLI scripts with argparse. Useful for testing tool logic outside the MCP server.

### Example Data (`examples/`)

Subtilisin protein variant dataset with `data.csv`, `wt.fasta`, `wt_struct.pdb`, and pre-computed embeddings. Used for testing all five tools.

## Key Dependencies (installed separately from requirements.txt)

- **PyTorch 2.4.0** with CUDA 11.8 (installed via conda)
- **ESM** (installed from `github.com/facebookresearch/esm`)
- **fastmcp** (MCP protocol framework)
- **torch_geometric** (graph neural network support for ESM-IF)

## Typical Protein Fitness Workflow

1. Extract embeddings → 2. Train regression model → 3. Predict fitness on new variants

Alternative zero-shot path (no training data needed): Calculate LLH scores directly.
