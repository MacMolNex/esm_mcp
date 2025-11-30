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
mamba env create -p ./env python=3.9 pip -y
mamba activate ./env
mamba install pytorch cudatoolkit=11.3 -c pytorch -y
mamba install pyg -c pyg -c conda-forge -y

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
