"""
ESM-based protein fitness prediction training tools.

This MCP Server provides 1 tool:
1. esm_train_fitness_model: Train regression models on ESM embeddings for fitness prediction

The tool performs 5-fold cross-validation and trains a final model on all data.
"""

# Standard imports
from typing import Annotated, Literal
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import joblib
import torch
from pathlib import Path
import os
from fastmcp import FastMCP
from datetime import datetime

# MCP server instance
esm_train_fitness_mcp = FastMCP(name="esm_train_fitness")


def load_data(data_dir, backbone_model='esm2_t33_650M_UR50D', target_col='log_fitness'):
    """Load ESM embedding data and target values."""
    data_file = os.path.join(data_dir, 'data.csv')
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    df_data = pd.read_csv(data_file)

    # Check if target column exists
    if target_col not in df_data.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV. Available columns: {df_data.columns.tolist()}")

    # Try to load pre-computed embeddings file
    embd_file = f'{data_file}.{backbone_model}.npy'

    if os.path.exists(embd_file):
        prot_embd = np.load(embd_file)
    else:
        # Load individual .pt files based on sequences.fasta
        layer = 36 if 't36' in backbone_model else 33

        fasta_file = os.path.join(data_dir, 'sequences.fasta')
        emb_dir = os.path.join(data_dir, backbone_model)

        if not os.path.exists(fasta_file):
            raise FileNotFoundError(f"FASTA file not found: {fasta_file}. Run esm_extract_embeddings_from_csv first with output_dir={data_dir}")

        if not os.path.isdir(emb_dir):
            # Check for common double-nesting issue
            nested = os.path.join(emb_dir, backbone_model)
            if os.path.isdir(nested):
                raise FileNotFoundError(
                    f"Embeddings directory has double-nesting: found {nested} instead of .pt files in {emb_dir}. "
                    f"Move the .pt files up one level: mv {nested}/*.pt {emb_dir}/"
                )
            raise FileNotFoundError(
                f"Embeddings directory not found: {emb_dir}. "
                f"Run esm_extract_embeddings_from_csv first with output_dir={data_dir}"
            )

        # Build seq -> embedding mapping
        seq2embd = {}
        seq_ids = []
        sequences = []

        with open(fasta_file, 'r') as f:
            current_id = None
            current_seq = []
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id is not None:
                        sequences.append(''.join(current_seq))
                        seq_ids.append(current_id)
                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_id is not None:
                sequences.append(''.join(current_seq))
                seq_ids.append(current_id)

        # Load embeddings
        for seq_id, seq in zip(seq_ids, sequences):
            emb_file = os.path.join(emb_dir, f'{seq_id}.pt')
            if not os.path.exists(emb_file):
                raise FileNotFoundError(f"Embedding file not found: {emb_file}")
            emb = torch.load(emb_file, map_location='cpu')['mean_representations'][layer]
            seq2embd[seq] = emb

        # Get embeddings for all sequences in data.csv
        prot_embd = []
        for idx, seq in enumerate(df_data['seq']):
            if seq in seq2embd:
                prot_embd.append(seq2embd[seq])
            else:
                raise ValueError(f"Could not find embeddings for sequence at index {idx}")

        prot_embd = torch.stack(prot_embd, dim=0).numpy()

        # Save for future use
        np.save(embd_file, prot_embd)

    Xs = prot_embd
    Ys = df_data[target_col].values

    return Xs, Ys


def apply_pca(Xs_train, Xs_test=None, n_components=60, output_dir=None, save_model=True):
    """Apply PCA transformation."""
    pca_model = PCA(n_components=n_components)
    Xs_train_pca = pca_model.fit_transform(Xs_train)

    if save_model and output_dir:
        pca_path = os.path.join(output_dir, 'pca_model.joblib')
        joblib.dump(pca_model, pca_path)

    if Xs_test is not None:
        Xs_test_pca = pca_model.transform(Xs_test)
        return Xs_train_pca, Xs_test_pca, pca_model
    else:
        return Xs_train_pca, None, pca_model


def create_reg_model(model_type='svr'):
    """Create regression model based on type."""
    if model_type == 'svm' or model_type == 'svr':
        from sklearn.svm import SVR
        model = SVR()
    elif model_type == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor()
    elif model_type == 'knn':
        from sklearn.neighbors import KNeighborsRegressor
        model = KNeighborsRegressor()
    elif model_type == 'gbdt':
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor()
    elif model_type == 'sgd':
        from sklearn.linear_model import SGDRegressor
        model = SGDRegressor()
    elif model_type == 'ridge':
        from sklearn.linear_model import Ridge
        model = Ridge()
    elif model_type == 'lasso':
        from sklearn.linear_model import Lasso
        model = Lasso()
    elif model_type == 'elastic_net':
        from sklearn.linear_model import ElasticNet
        model = ElasticNet()
    elif model_type == 'mlp':
        from sklearn.neural_network import MLPRegressor
        model = MLPRegressor()
    elif model_type == 'xgboost':
        try:
            import xgboost as xgb
            model = xgb.XGBRegressor()
        except ImportError:
            raise ImportError("XGBoost not installed. Please install with: pip install xgboost")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def perform_cross_validation(Xs, Ys, backbone_model, head_model, n_components, seed, output_dir):
    """Perform 5-fold cross validation."""
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = []
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(Xs)):
        # Split data
        Xs_train_fold = Xs[train_idx]
        Xs_test_fold = Xs[test_idx]
        ys_train_fold = Ys[train_idx]
        ys_test_fold = Ys[test_idx]

        # Create fold directory
        fold_output_dir = os.path.join(output_dir, 'cv_folds', f'fold_{fold + 1}')
        Path(fold_output_dir).mkdir(parents=True, exist_ok=True)

        # Apply PCA
        Xs_train_fold_pca, Xs_test_fold_pca, _ = apply_pca(
            Xs_train_fold, Xs_test_fold,
            n_components=n_components,
            output_dir=fold_output_dir,
            save_model=True
        )

        # Train model
        model = create_reg_model(head_model)
        model.fit(Xs_train_fold_pca, ys_train_fold)

        # Make predictions
        assert Xs_test_fold_pca is not None
        ys_test_pred_fold = model.predict(Xs_test_fold_pca)
        ys_train_pred_fold = model.predict(Xs_train_fold_pca)

        # Calculate metrics
        spearman_r_test = spearmanr(ys_test_fold, ys_test_pred_fold)[0]
        spearman_r_train = spearmanr(ys_train_fold, ys_train_pred_fold)[0]

        cv_scores.append(spearman_r_test)

        # Save fold model
        joblib.dump(model, os.path.join(fold_output_dir, f'head_model_{head_model}.joblib'))
        np.save(os.path.join(fold_output_dir, 'ys_test_pred.npy'), ys_test_pred_fold)
        np.save(os.path.join(fold_output_dir, 'ys_test.npy'), ys_test_fold)

        fold_results.append({
            'fold': fold + 1,
            'train_spearman': float(spearman_r_train),
            'test_spearman': float(spearman_r_test),
            'train_size': int(len(ys_train_fold)),
            'test_size': int(len(ys_test_fold))
        })

    return cv_scores, fold_results


def train_final_model(Xs, Ys, head_model, n_components, output_dir):
    """Train final model on all data."""
    final_output_dir = os.path.join(output_dir, 'final_model')
    Path(final_output_dir).mkdir(parents=True, exist_ok=True)

    # Apply PCA
    Xs_pca, _, pca_model = apply_pca(
        Xs,
        n_components=n_components,
        output_dir=final_output_dir,
        save_model=True
    )

    # Train final model
    final_model = create_reg_model(head_model)
    final_model.fit(Xs_pca, Ys)

    # Make predictions
    Ys_pred = final_model.predict(Xs_pca)

    # Calculate training Spearman
    train_spearman = spearmanr(Ys, Ys_pred)[0]

    # Save final model
    model_path = os.path.join(final_output_dir, f'head_model_{head_model}.joblib')
    joblib.dump(final_model, model_path)

    # Save predictions
    np.save(os.path.join(final_output_dir, 'ys_train_pred.npy'), Ys_pred)
    np.save(os.path.join(final_output_dir, 'ys_train.npy'), Ys)

    return final_model, pca_model, train_spearman


@esm_train_fitness_mcp.tool
def esm_train_fitness_model(
    input_dir: Annotated[str, "Input directory containing data.csv and embeddings"],
    output_dir: Annotated[str, "Output directory for models and results"],
    backbone_model: Annotated[
        Literal[
            "esm2_t33_650M_UR50D",
            "esm1v_t33_650M_UR90S_1",
            "esm1v_t33_650M_UR90S_2",
            "esm1v_t33_650M_UR90S_3",
            "esm1v_t33_650M_UR90S_4",
            "esm1v_t33_650M_UR90S_5",
            "esm2_t36_3B_UR50D"
        ],
        "ESM backbone model"
    ] = "esm2_t33_650M_UR50D",
    head_model: Annotated[
        Literal['svr', 'svm', 'random_forest', 'knn', 'gbdt', 'sgd', 'ridge', 'lasso', 'elastic_net', 'mlp', 'xgboost'],
        "Regression head model"
    ] = "svr",
    n_components: Annotated[int, "Number of PCA components"] = 60,
    seed: Annotated[int, "Random seed for reproducibility"] = 42,
    target_col: Annotated[str, "Target column name in CSV"] = "log_fitness",
    no_final_model: Annotated[bool, "Skip training final model on all data"] = False,
) -> dict:
    """
    Train ESM-based fitness prediction models with 5-fold cross-validation.

    This tool:
    1. Loads ESM embeddings and target values from data directory
    2. Performs 5-fold cross-validation with PCA and regression model
    3. Trains final model on all data (unless skipped)
    4. Saves models, predictions, and training statistics
    5. Returns comprehensive training results

    Input: Data directory with embeddings, output directory, model parameters
    Output: Dictionary with CV results, final model performance, and file paths
    """
    try:
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Load data
        Xs, Ys = load_data(input_dir, backbone_model, target_col)

        # Perform cross validation
        cv_scores, fold_results = perform_cross_validation(
            Xs, Ys, backbone_model, head_model, n_components, seed, output_dir
        )

        # Calculate CV statistics
        mean_cv_score = float(np.mean(cv_scores))
        std_cv_score = float(np.std(cv_scores))
        min_cv_score = float(np.min(cv_scores))
        max_cv_score = float(np.max(cv_scores))

        # Save CV results
        df_cv_results = pd.DataFrame(fold_results)
        cv_results_path = os.path.join(output_dir, f'{backbone_model}_{head_model}_cv_results.csv')
        df_cv_results.to_csv(cv_results_path, index=False)

        # Prepare summary
        summary = {
            'backbone_model': backbone_model,
            'head_model': head_model,
            'n_components': n_components,
            'seed': seed,
            'n_samples': int(len(Xs)),
            'n_features': int(Xs.shape[1]),
            'mean_cv_spearman': mean_cv_score,
            'std_cv_spearman': std_cv_score,
            'min_cv_spearman': min_cv_score,
            'max_cv_spearman': max_cv_score
        }

        # Train final model
        final_model_info = None
        if not no_final_model:
            final_model, pca_model, train_spearman = train_final_model(
                Xs, Ys, head_model, n_components, output_dir
            )
            summary['final_train_spearman'] = float(train_spearman)
            final_model_info = {
                "train_spearman": float(train_spearman),
                "model_path": os.path.join(output_dir, 'final_model', f'head_model_{head_model}.joblib'),
                "pca_path": os.path.join(output_dir, 'final_model', 'pca_model.joblib'),
            }

        # Save summary
        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(output_dir, 'training_summary.csv')
        summary_df.to_csv(summary_path, index=False)

        return {
            "status": "success",
            "output_dir": output_dir,
            "backbone_model": backbone_model,
            "head_model": head_model,
            "n_components": n_components,
            "n_samples": int(len(Xs)),
            "cross_validation": {
                "mean_spearman": mean_cv_score,
                "std_spearman": std_cv_score,
                "min_spearman": min_cv_score,
                "max_spearman": max_cv_score,
                "fold_scores": [float(s) for s in cv_scores],
                "fold_results": fold_results,
                "results_file": cv_results_path,
            },
            "final_model": final_model_info,
            "summary_file": summary_path,
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "input_dir": input_dir,
            "output_dir": output_dir,
        }
