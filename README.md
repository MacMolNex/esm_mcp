# ESM MCP server

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

```bash
# Create and activate virtual environment
mamba env create -p ./env python=3.10 pip -y
mamba activate ./env
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install biotite==0.36.1
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
pip install torch_geometric

pip install git+https://github.com/facebookresearch/esm.git
pip install biotite pandas biopython numpy==1.26.4

# Install dependencies
pip install -r requirements.txt
pip install --ignore-installed fastmcp
```

## Local usage

### 1. Extracting ESM embeddings from CSV

```shell
python notebooks/esm_embedding.py -i examples/data.csv -m esm2_t33_650M_UR50D
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
python notebooks/esm_train_fitness.py -i examples/ -o examples/esm_fitness -b esm2_t33_650M_UR50D -m svr
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
python notebooks/esm_predict_fitness.py  -i  examples/data.csv  -m examples/esm2_650M_RF/final_model -b esm2_t33_650M_UR50D
```

### 4. ESM likelihood calculation for variants
In cases where not enough variant data is available, a good solutions is to use likelihoods to evaluate the fitness of variants.

```shell
python notebooks/esm_llh.py -i examples/data.csv -w examples/wt.fasta -m esm2_t33_650M_UR50D
```
### 5. ESM-IF likelihood calculation with structure
In cases where not enough variant data is available but a good structure of wild-type is available, another good solutions is to use ESM-IF likelihoods to evaluate the fitness of variants.

```shell
python notebooks/esm_if_llh.py -i examples/data.csv -w examples/wt.fasta -p examples/wt_struct.pdb
```

## MCP usage

### Install ESM MCP server
```shell
fastmcp install claude-code mcp-servers/esm_mcp/src/esm_mcp.py --python mcp-servers/esm_mcp/env/bin/python
```

### Available MCP Tools

The ESM MCP server provides 5 tools that correspond to the 5 notebook scripts:

#### 1. `esm_extract_embeddings_from_csv`
Extract ESM embeddings from a CSV file containing protein sequences.
- **Input**: CSV file path, ESM model name, sequence column name
- **Output**: FASTA file, embeddings directory (.pt files), statistics
- **Use case**: First step for supervised learning workflows

#### 2. `esm_calculate_llh`
Calculate ESM log-likelihood for protein mutations (sequence-based, zero-shot).
- **Input**: CSV with sequences, wild-type FASTA, ESM model name
- **Output**: CSV with LLH scores, statistics, optional correlation with fitness
- **Use case**: Zero-shot fitness prediction without training data

#### 3. `esm_if_calculate_llh`
Calculate ESM-IF log-likelihood using protein structure (structure-based, zero-shot).
- **Input**: CSV with sequences, wild-type FASTA, PDB structure file, chain ID
- **Output**: CSV with LLH scores, statistics, optional correlation with fitness
- **Use case**: Zero-shot fitness prediction with structural information

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
    model_name="esm2_t33_650M_UR50D"
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
    model_name="esm2_t33_650M_UR50D"
)

# Alternative: Zero-shot prediction with structure-based LLH
if_llh_result = esm_if_calculate_llh(
    data_csv="data.csv",
    wt_fasta="wt.fasta",
    pdb_file="wt_struct.pdb",
    chain="A"
)
```
