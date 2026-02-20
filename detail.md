# ESM MCP server - Detailed Documentation

ESM MCP server for protein modeling, extracted from the official ESM tutorial.


## Overview

This ESM MCP server provides comprehensive protein structure analysis tools using ESM (Evolutionary Scale Modeling) models. Here we have 5 main scripts for comprehensive protein analysis:

- 1. Extract embeddings (supervised learning)
- 2. Train fitness model (supervised)
- 3. Predict fitness (supervised)
- 4. Calculate sequence-based LLH (zero-shot)
- 5. Calculate structure-based LLH (zero-shot)

Note: ESM-Fold MCP is created in another mcp server called `esmfold_mcp` as it has some special dependencies.

## Installation

### Option 1: Docker (Recommended)

The easiest way to get started. A pre-built image is published to GHCR on every push to `main`.

```bash
# Pull the latest image
docker pull ghcr.io/macromnex/esm_mcp:latest

# Run the MCP server
docker run --gpus all -p 8000:8000 ghcr.io/macromnex/esm_mcp:latest

# Or build locally
docker build -t esm_mcp .
docker run --gpus all -p 8000:8000 esm_mcp
```

Register in Claude Code:
```bash
claude mcp add esm -- docker run --gpus all -i --rm ghcr.io/macromnex/esm_mcp:latest
```

### Option 2: Clone + Download Pre-built Environment (for Colab / fast setup)

Downloads a pre-packaged conda environment from GitHub Releases instead of building from scratch. Useful for Google Colab or machines without conda configured.

```bash
git clone https://github.com/MacromNex/esm_mcp.git
cd esm_mcp
USE_PACKED_ENVS=1 bash quick_setup.sh
```

For Google Colab:
```python
import subprocess, os
subprocess.run(["git", "clone", "https://github.com/MacromNex/esm_mcp.git"])
os.chdir("esm_mcp")
subprocess.run(["bash", "-c", "USE_PACKED_ENVS=1 bash quick_setup.sh"])
```

### Option 3: Clone + Create Environment from Scratch

Full setup that creates a fresh conda environment and installs all dependencies. Requires conda or mamba.

```bash
git clone https://github.com/MacromNex/esm_mcp.git
cd esm_mcp
bash quick_setup.sh
```

The script will create the conda environment, install all dependencies, clone the ESM repository, and display the Claude Code configuration. See `quick_setup.sh --help` for options like `--skip-env` or `--skip-repo`.

After setup, register in Claude Code:
```bash
claude mcp add esm -- ./env/bin/python src/server.py
```

## Local usage

### 1. Extracting ESM embeddings from CSV

```shell
python scripts/esm_embedding.py -i examples/data.csv -m esm2_t33_650M_UR50D
```

**Supported ESM Models:**
- `esm2_t33_650M_UR50D` (default, 650M parameters, layer 33)
- `esm1v_t33_650M_UR90S_1` through `esm1v_t33_650M_UR90S_5` (650M parameters, layer 33)
- `esm2_t36_3B_UR50D` (3B parameters, layer 36)
Same ESM models applies for fitness modeling and ESM likelihood calculation.

### 2. Training fitness Models with extracted embeddings

After extracting embeddings, use them for fitness prediction:

```shell
# Train with ESM embeddings (5-fold cross validation)
python scripts/esm_train_fitness.py -i examples/ -o examples/esm_fitness -b esm2_t33_650M_UR50D -m svr
```

**Supported Head Models:**
- `svm` / `svr`: Support Vector Machine Regression
- `random_forest`: Random Forest Regressor
- `knn`: K-Nearest Neighbors Regressor
- `gbdt`: Gradient Boosting Decision Tree Regressor
- `sgd`: Stochastic Gradient Descent Regressor
- `ridge`: Ridge Regression
- `mlp`: Multi-Layer Perceptron Regressor
- `xgboost`: XGBoost Regressor (requires xgboost package)
- `lasso`: Lasso Regression
- `elastic_net`: Elastic Net Regression

### 3. Predict with a fitness model using extracted embeddings

```shell
python scripts/esm_predict_fitness.py  -i  examples/data.csv  -m examples/esm2_650M_RF/final_model -b esm2_t33_650M_UR50D
```

### 4. ESM likelihood calculation for variants
In cases where not enough variant data is available, a good solutions is to use likelihoods to evaluate the fitness of variants.

```shell
python scripts/esm_llh.py -i examples/data.csv -w examples/wt.fasta -m esm2_t33_650M_UR50D
```
### 5. ESM-IF likelihood calculation with structure
In cases where not enough variant data is available but a good structure of wild-type is available, another good solutions is to use ESM-IF likelihoods to evaluate the fitness of variants.

```shell
python scripts/esm_if_llh.py -i examples/data.csv -w examples/wt.fasta -p examples/wt_struct.pdb
```

## MCP usage

### Install ESM MCP server
```shell
fastmcp install claude-code tool-mcps/esm_mcp/src/server.py --python tool-mcps/esm_mcp/env/bin/python
fastmcp install gemini-cli tool-mcps/esm_mcp/src/server.py --python tool-mcps/esm_mcp/env/bin/python

```

### Call ESM MCP service
1. Train a ESM-fitness model
```markdown
Can you help train a esm model for data @examples/ and save it to @examples/esm_fitness using the esm mcp server?
Please convert the relative path to absolution path before calling the MCP servers.
Obtain the embeddings if it is not created.
```
2. Inference ESM likelihoods
```markdown
Can you help intererence esm likelihood for data @examples/case2.1_subtilisin/data.csv with wild-type  @examples/case2.1_subtilisin/wt.fasta using the esm_llh_mcp api in esm mcp server. Please write the output to @examples/case2.1_subtilisin/data.csv_esm_llh.csv

This MCP service finish in seconds.
Please convert the relative path to absolution path before calling the MCP servers.
Please use cuda device 1 where available.
```

3. Inference ESM-IF likelihoods
```markdown
Can you help intererence esm-if likelihood for data @examples/case2.1_subtilisin/data.csv with wild-type sequence  @examples/case2.1_subtilisin/wt.fasta and structure @examples/case2.1_subtilisin/wt_struct.pdb using the esm_if_llh api in esm mcp server. Please write the output to @examples/case2.1_subtilisin/data.csv_esmif_llh.csv

Please convert the relative path to absolution path before calling the MCP servers.
Please use cuda device 1 where available.
```

### Available MCP Tools

The ESM MCP server provides 5 tools that correspond to the 5 notebook scripts:

#### 1. `esm_extract_embeddings_from_csv`
Extract ESM embeddings from a CSV file containing protein sequences.
- **Input**: CSV file path, ESM model name, sequence column name, device (optional)
- **Output**: FASTA file, embeddings directory (.pt files), statistics
- **Use case**: First step for supervised learning workflows
- **Device support**: Supports `cuda`, `cuda:0`, `cuda:1`, `cpu` formats

#### 2. `esm_calculate_llh`
Calculate ESM log-likelihood for protein mutations (sequence-based, zero-shot).
- **Input**: CSV with sequences, wild-type FASTA, ESM model name, device (optional)
- **Output**: CSV with LLH scores, statistics, optional correlation with fitness
- **Use case**: Zero-shot fitness prediction without training data
- **Device support**: Supports `cuda`, `cuda:0`, `cuda:1`, `cpu` formats

#### 3. `esm_if_calculate_llh`
Calculate ESM-IF log-likelihood using protein structure (structure-based, zero-shot).
- **Input**: CSV with sequences, wild-type FASTA, PDB structure file, chain ID, device (optional)
- **Output**: CSV with LLH scores, statistics, optional correlation with fitness
- **Use case**: Zero-shot fitness prediction with structural information
- **Device support**: Supports `cuda`, `cuda:0`, `cuda:1`, `cpu` formats

#### 4. `esm_train_fitness_model`
Train regression models on ESM embeddings for fitness prediction (supervised).
- **Input**: Data directory with embeddings, ESM model name, head model type, PCA components
- **Output**: Trained models (PCA + head model), cross-validation results, training statistics
- **Use case**: Train predictive models when training data is available

#### 5. `esm_predict_fitness`
Predict fitness values using pre-trained ESM-based models.
- **Input**: CSV with sequences, trained model directory, ESM model name
- **Output**: CSV with predicted fitness values, prediction statistics
- **Use case**: Make predictions on new sequences using trained models

### Example MCP Workflow

```python
# 1. Extract embeddings
result = esm_extract_embeddings_from_csv(
    csv_path="data.csv",
    model_name="esm2_t33_650M_UR50D",
    device="cuda:0"  # Specify GPU device
)

# 2. Train fitness model
train_result = esm_train_fitness_model(
    input_dir="data/",
    output_dir="models/esm_fitness",
    backbone_model="esm2_t33_650M_UR50D",
    head_model="svr",
    n_components=60
)

# 3. Predict fitness for new sequences
pred_result = esm_predict_fitness(
    data_csv="new_data.csv",
    model_dir="models/esm_fitness/final_model",
    backbone_model="esm2_t33_650M_UR50D"
)

# Alternative: Zero-shot prediction with sequence-based LLH
llh_result = esm_calculate_llh(
    data_csv="data.csv",
    wt_fasta="wt.fasta",
    model_name="esm2_t33_650M_UR50D",
    device="cuda:1"  # Use specific GPU
)

# Alternative: Zero-shot prediction with structure-based LLH
if_llh_result = esm_if_calculate_llh(
    data_csv="data.csv",
    wt_fasta="wt.fasta",
    pdb_file="wt_struct.pdb",
    chain="A",
    device="cuda:1"  # Use specific GPU
)
```

### Device Selection

All ESM tools now support explicit device specification:
- **Format**: `cuda`, `cuda:0`, `cuda:1`, `cpu`
- **Default**: Uses `cuda` (first available GPU) if not specified
- **Multi-GPU**: Use `cuda:0`, `cuda:1`, etc. to select specific GPUs
- **CPU only**: Use `cpu` to force CPU execution
