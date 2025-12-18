# ESM Embedding and Fitness Prediction Examples

This directory contains example data and usage instructions for ESM embedding extraction and fitness prediction workflows.

## Quick Start

### Basic Usage

Extract embeddings using the default model (esm2_t33_650M_UR50D):

```bash
python ../notebooks/esm_embedding.py -i sample_proteins.csv
```

This will create:
- `sequences.fasta` - FASTA file with all unique sequences
- `esm2_t33_650M_UR50D/` - Directory containing the extracted embeddings

### Using a Different Model

```bash
python ../notebooks/esm_embedding.py -i sample_proteins.csv -m esm2_t36_3B_UR50D
```

### Custom Sequence Column

If your CSV has a different column name for sequences:

```bash
python ../notebooks/esm_embedding.py -i sample_proteins.csv -s sequence_column
```

### Using Custom IDs

To use a specific column for sequence IDs instead of auto-generated seq_0, seq_1, etc.:

```bash
python ../notebooks/esm_embedding.py -i sample_proteins.csv -d protein_id
```

## Available Models

- `esm2_t33_650M_UR50D` (default) - ESM-2 650M parameters
- `esm1v_t33_650M_UR90S_1` - ESM-1v variant 1
- `esm1v_t33_650M_UR90S_2` - ESM-1v variant 2
- `esm1v_t33_650M_UR90S_3` - ESM-1v variant 3
- `esm1v_t33_650M_UR90S_4` - ESM-1v variant 4
- `esm1v_t33_650M_UR90S_5` - ESM-1v variant 5
- `esm2_t36_3B_UR50D` - ESM-2 3B parameters (larger model)

## CSV File Format

Your CSV file should have at least one column containing protein sequences. Example:

```csv
seq,protein_id,description
MKTIIALSYIFCLVFA,prot_001,Sample protein 1
ARNDCQEGHILKMFPS,prot_002,Sample protein 2
```

Required:
- A column with protein sequences (default name: "seq")

Optional:
- A column with sequence IDs (if not provided, auto-generates seq_0, seq_1, etc.)
- Any other metadata columns

## Output Structure

After running the script, you'll find in the same directory as your CSV:

```
your_directory/
├── your_data.csv          # Your original CSV file
├── sequences.fasta        # Generated FASTA file
└── esm2_t33_650M_UR50D/  # Embeddings directory (named after model)
    ├── seq_0.pt          # Embedding for sequence 0
    ├── seq_1.pt          # Embedding for sequence 1
    └── ...
```

## Complete Fitness Prediction Workflow

For protein fitness prediction, follow this two-step workflow:

### Quick Start with Example Data

Run the complete workflow with the example script:

```bash
# Make sure dependencies are installed first
# pip install -r ../requirements.txt

# Run the complete workflow
./example_workflow.sh
```

### Manual Workflow

#### Step 1: Extract Embeddings

```bash
python ../notebooks/esm_embedding.py \
    -i sample_fitness_data.csv \
    -m esm2_t33_650M_UR50D \
    -d protein_id
```

#### Step 2: Train Fitness Model

```bash
python ../notebooks/esm_train_fitness.py \
    -i . \
    -o results \
    -b esm2_t33_650M_UR50D \
    -m svr \
    -n 60
```

This will:
1. Run 5-fold cross-validation to evaluate performance
2. Train a final model on all data
3. Save models and results to the `results/` directory

### Understanding the Output

After training, check:
- `results/training_summary.csv` - Overall performance metrics
- `results/final_model/head_model_svr.joblib` - Trained model for predictions
- `results/esm2_t33_650M_UR50D_svr_cv_results.csv` - Detailed CV results

### Example Files

- `sample_proteins.csv` - Simple protein sequences for embedding extraction
- `sample_fitness_data.csv` - Protein sequences with fitness values for training
- `example_workflow.sh` - Complete automated workflow script

## Help

For full help and all options:

```bash
python ../notebooks/esm_embedding.py --help
python ../notebooks/esm_train_fitness.py --help
```

## Making Predictions on New Data

After training a model, you can use it to predict fitness for new sequences:

### Prediction Workflow

#### Step 1: Extract embeddings for new data

```bash
python ../notebooks/esm_embedding.py \
    -i new_data.csv \
    -m esm2_t33_650M_UR50D
```

#### Step 2: Predict fitness

```bash
python ../notebooks/esm_predict_fitness.py \
    -i new_data.csv \
    -m esm_fitness/final_model \
    -b esm2_t33_650M_UR50D
```

This creates `new_data_esm_fitness_pred.csv` with predictions.

### Quick Prediction Example

```bash
# Run the automated prediction workflow
./example_predict_workflow.sh
```

## Calculating ESM Log-Likelihood

For mutation effect prediction using ESM log-likelihood:

### Sequence-Based LLH Workflow

```bash
python ../notebooks/esm_llh.py \
    -i data.csv \
    -w wt.fasta \
    -m esm2_t33_650M_UR50D
```

This calculates log-likelihood scores based on sequence only.

### Structure-Based LLH Workflow (ESM-IF)

```bash
python ../notebooks/esm_if_llh.py \
    -i data.csv \
    -w wt.fasta \
    -p wt_struct.pdb \
    -c A
```

This calculates log-likelihood scores using protein structure information.

### Quick LLH Examples

```bash
# Run sequence-based LLH
./example_llh_workflow.sh

# Run structure-based LLH
./example_esmif_workflow.sh
```

## Predicting Protein Structures with ESMFold

ESMFold predicts 3D protein structures from sequences without requiring MSAs.

### Structure Prediction Workflow

```bash
python ../notebooks/esm_fold.py \
    -i sequences.fasta \
    -o predicted_structures/
```

This predicts structures and saves PDB files with confidence scores.

### Quick ESMFold Example

```bash
# Run the automated structure prediction workflow
./example_esmfold_workflow.sh
```

### Output Interpretation

Each PDB file contains:
- 3D atomic coordinates
- pLDDT scores (per-residue confidence) in B-factor column
- pTM score reported in console (overall structure quality)

For detailed documentation, see:
- `../notebooks/README_training.md` - Complete training guide
- `../notebooks/README_prediction.md` - Prediction guide
- `../notebooks/README_llh.md` - Log-likelihood calculation guide
- `../notebooks/README_esmif.md` - ESM-IF structure-based scoring guide
- `../notebooks/README_esmfold.md` - ESMFold structure prediction guide
