#!/usr/bin/env python3
"""
ESM-based Protein Fitness Prediction Training Script

This script trains regression models on ESM embeddings for protein fitness prediction.
It performs 5-fold cross-validation for evaluation and then trains a final model on all data.

Usage:
    python esm_train_fitness.py -i <data_dir> -o <output_dir> -b <esm_model>

Example:
    python esm_train_fitness.py -i data/proteins -o results -b esm2_t33_650M_UR50D
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import joblib
import torch
from loguru import logger


def load_data(data_dir, backbone_model='esm2_t33_650M_UR50D', target_col='log_fitness'):
    """
    Load ESM embedding data and target values.

    Args:
        data_dir: Directory containing data.csv and embeddings
        backbone_model: ESM model name
        target_col: Target column name in CSV

    Returns:
        Xs: Feature matrix (embeddings)
        Ys: Target values
    """
    data_file = os.path.join(data_dir, 'data.csv')
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    df_data = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df_data)} samples from {data_file}")

    # Check if target column exists
    if target_col not in df_data.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV. Available columns: {df_data.columns.tolist()}")

    # Try to load pre-computed embeddings file
    embd_file = f'{data_file}.{backbone_model}.npy'

    if os.path.exists(embd_file):
        logger.info(f"Loading pre-computed embeddings from {embd_file}")
        prot_embd = np.load(embd_file)
    else:
        # Load individual .pt files based on sequences.fasta
        logger.info(f"Loading embeddings from individual .pt files...")

        # Determine layer based on model
        layer = 36 if 't36' in backbone_model else 33

        # Check for sequences.fasta to get seq_id mapping
        fasta_file = os.path.join(data_dir, 'sequences.fasta')
        emb_dir = os.path.join(data_dir, backbone_model)

        if not os.path.exists(fasta_file):
            raise FileNotFoundError(f"FASTA file not found: {fasta_file}. Cannot map sequences to embedding files.")

        # Build seq -> embedding mapping from FASTA and .pt files
        logger.info(f"Building sequence to embedding mapping from {fasta_file}")
        seq2embd = {}
        seq_ids = []
        sequences = []

        # Parse FASTA file to get seq_id -> sequence mapping
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
            # Don't forget the last sequence
            if current_id is not None:
                sequences.append(''.join(current_seq))
                seq_ids.append(current_id)

        logger.info(f"Found {len(seq_ids)} sequences in FASTA file")

        # Load embeddings and build seq -> embedding mapping
        for seq_id, seq in zip(seq_ids, sequences):
            emb_file = os.path.join(emb_dir, f'{seq_id}.pt')
            if not os.path.exists(emb_file):
                raise FileNotFoundError(f"Embedding file not found: {emb_file}")
            emb = torch.load(emb_file)['mean_representations'][layer]
            seq2embd[seq] = emb

        logger.info(f"Loaded {len(seq2embd)} embeddings")

        # Now get embeddings for all sequences in data.csv in order
        prot_embd = []
        missing_seqs = []

        for idx, seq in enumerate(df_data['seq']):
            if seq in seq2embd:
                prot_embd.append(seq2embd[seq])
            else:
                raise ValueError(f"Could not find embeddings for sequence {seq} at index {idx} in data.csv")

        prot_embd = torch.stack(prot_embd, dim=0).numpy()

        # Save for future use
        logger.info(f"Saving combined embeddings to {embd_file}")
        np.save(embd_file, prot_embd)

    Xs = prot_embd
    Ys = df_data[target_col].values

    logger.info(f"Feature shape: {Xs.shape}, Target shape: {Ys.shape}")

    return Xs, Ys


def apply_pca(Xs_train, Xs_test=None, n_components=60, output_dir=None, save_model=True):
    """
    Apply PCA transformation to training and test data.

    Args:
        Xs_train: Training features
        Xs_test: Test features (optional)
        n_components: Number of PCA components
        output_dir: Directory to save PCA model
        save_model: Whether to save the PCA model

    Returns:
        Transformed training data, transformed test data (if provided), PCA model
    """
    pca_model = PCA(n_components=n_components)
    Xs_train_pca = pca_model.fit_transform(Xs_train)

    explained_var = np.sum(pca_model.explained_variance_ratio_)
    logger.info(f"PCA: {n_components} components explain {explained_var:.3%} of variance")

    if save_model and output_dir:
        pca_path = os.path.join(output_dir, 'pca_model.joblib')
        joblib.dump(pca_model, pca_path)
        logger.debug(f"PCA model saved to {pca_path}")

    if Xs_test is not None:
        Xs_test_pca = pca_model.transform(Xs_test)
        return Xs_train_pca, Xs_test_pca, pca_model
    else:
        return Xs_train_pca, None, pca_model


def create_reg_model(model_type='svm'):
    """
    Create regression model based on type.

    Args:
        model_type: Type of regression model

    Returns:
        Initialized model
    """
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


def perform_cross_validation(Xs, Ys, args):
    """
    Perform 5-fold cross validation.

    Args:
        Xs: Feature matrix
        Ys: Target values
        args: Command line arguments

    Returns:
        cv_scores: List of Spearman correlations for each fold
        fold_results: Detailed results for each fold
    """
    logger.info("Starting 5-fold cross validation...")

    # Setup 5-fold cross validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_scores = []
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(Xs)):
        logger.info(f"\n--- Fold {fold + 1}/5 ---")

        # Split data for this fold
        Xs_train_fold = Xs[train_idx]
        Xs_test_fold = Xs[test_idx]
        ys_train_fold = Ys[train_idx]
        ys_test_fold = Ys[test_idx]

        # Create fold-specific output directory
        fold_output_dir = os.path.join(args.output_dir, 'cv_folds', f'fold_{fold + 1}')
        Path(fold_output_dir).mkdir(parents=True, exist_ok=True)

        # Apply PCA
        Xs_train_fold_pca, Xs_test_fold_pca, _ = apply_pca(
            Xs_train_fold, Xs_test_fold,
            n_components=args.n_components,
            output_dir=fold_output_dir,
            save_model=True
        )

        # Create and train model for this fold
        head_model = create_reg_model(args.head_model)
        head_model.fit(Xs_train_fold_pca, ys_train_fold)

        # Make predictions
        # Type assertion: Xs_test_fold_pca is guaranteed to be ndarray, not None
        assert Xs_test_fold_pca is not None
        ys_test_pred_fold = head_model.predict(Xs_test_fold_pca)
        ys_train_pred_fold = head_model.predict(Xs_train_fold_pca)

        # Calculate metrics
        spearman_r_test = spearmanr(ys_test_fold, ys_test_pred_fold)[0]
        spearman_r_train = spearmanr(ys_train_fold, ys_train_pred_fold)[0]

        cv_scores.append(spearman_r_test)

        logger.info(f"Train Spearman: {spearman_r_train:.4f}, Test Spearman: {spearman_r_test:.4f}")

        # Save fold model and predictions
        joblib.dump(head_model, os.path.join(fold_output_dir, f'head_model_{args.head_model}.joblib'))
        np.save(os.path.join(fold_output_dir, 'ys_test_pred.npy'), ys_test_pred_fold)
        np.save(os.path.join(fold_output_dir, 'ys_test.npy'), ys_test_fold)

        # Store fold results
        fold_results.append({
            'fold': fold + 1,
            'train_spearman': spearman_r_train,
            'test_spearman': spearman_r_test,
            'train_size': len(ys_train_fold),
            'test_size': len(ys_test_fold)
        })

    return cv_scores, fold_results


def train_final_model(Xs, Ys, args):
    """
    Train final model on all data.

    Args:
        Xs: Feature matrix
        Ys: Target values
        args: Command line arguments

    Returns:
        final_model: Trained model
        pca_model: Fitted PCA model
        train_spearman: Training Spearman correlation
    """
    logger.info("\nTraining final model on all data...")

    # Create final model output directory
    final_output_dir = os.path.join(args.output_dir, 'final_model')
    Path(final_output_dir).mkdir(parents=True, exist_ok=True)

    # Apply PCA on all data
    Xs_pca, _, pca_model = apply_pca(
        Xs,
        n_components=args.n_components,
        output_dir=final_output_dir,
        save_model=True
    )

    # Create and train final model
    final_model = create_reg_model(args.head_model)
    final_model.fit(Xs_pca, Ys)

    # Make predictions on training data
    Ys_pred = final_model.predict(Xs_pca)

    # Calculate training Spearman correlation
    train_spearman = spearmanr(Ys, Ys_pred)[0]
    logger.info(f"Final model training Spearman: {train_spearman:.4f}")

    # Save final model
    model_path = os.path.join(final_output_dir, f'head_model_{args.head_model}.joblib')
    joblib.dump(final_model, model_path)
    logger.info(f"Final model saved to {model_path}")

    # Save predictions
    np.save(os.path.join(final_output_dir, 'ys_train_pred.npy'), Ys_pred)
    np.save(os.path.join(final_output_dir, 'ys_train.npy'), Ys)

    return final_model, pca_model, train_spearman


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ESM-based protein fitness prediction with 5-fold CV and final model training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings (ESM2 650M, SVR, PCA 60)
  python esm_train_fitness.py -i data/proteins -o results

  # Use larger ESM model
  python esm_train_fitness.py -i data/proteins -o results -b esm2_t36_3B_UR50D

  # Use different head model
  python esm_train_fitness.py -i data/proteins -o results -m random_forest

  # Custom PCA components
  python esm_train_fitness.py -i data/proteins -o results -n 100

Available ESM models:
  - esm2_t33_650M_UR50D (default)
  - esm1v_t33_650M_UR90S_1
  - esm1v_t33_650M_UR90S_2
  - esm1v_t33_650M_UR90S_3
  - esm1v_t33_650M_UR90S_4
  - esm1v_t33_650M_UR90S_5
  - esm2_t36_3B_UR50D

Available head models:
  - svr (default)
  - random_forest
  - knn
  - gbdt
  - ridge
  - lasso
  - elastic_net
  - mlp
  - xgboost
        """
    )

    parser.add_argument(
        '-i', '--input_dir',
        type=str,
        required=True,
        help='Input directory containing data.csv and embeddings'
    )
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        required=True,
        help='Output directory for models and results'
    )
    parser.add_argument(
        '-b', '--backbone_model',
        type=str,
        default='esm2_t33_650M_UR50D',
        choices=[
            'esm2_t33_650M_UR50D',
            'esm1v_t33_650M_UR90S_1',
            'esm1v_t33_650M_UR90S_2',
            'esm1v_t33_650M_UR90S_3',
            'esm1v_t33_650M_UR90S_4',
            'esm1v_t33_650M_UR90S_5',
            'esm2_t36_3B_UR50D'
        ],
        help='ESM backbone model (default: esm2_t33_650M_UR50D)'
    )
    parser.add_argument(
        '-m', '--head_model',
        type=str,
        default='svr',
        choices=['svr', 'svm', 'random_forest', 'knn', 'gbdt', 'sgd', 'ridge', 'lasso', 'elastic_net', 'mlp', 'xgboost'],
        help='Regression head model (default: svr)'
    )
    parser.add_argument(
        '-n', '--n_components',
        type=int,
        default=60,
        help='Number of PCA components (default: 60)'
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--target_col',
        type=str,
        default='log_fitness',
        help='Target column name in CSV (default: log_fitness)'
    )
    parser.add_argument(
        '--no_final_model',
        action='store_true',
        help='Skip training final model on all data (only do CV)'
    )

    return parser.parse_args()


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main training pipeline."""
    args = get_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Set random seed for reproducibility
    set_random_seed(args.seed)

    # Setup logging
    log_file = os.path.join(args.output_dir, 'training.log')
    logger.add(log_file, rotation="10 MB")

    # Print configuration
    logger.info("=" * 80)
    logger.info("ESM Fitness Prediction Training")
    logger.info("=" * 80)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Backbone model: {args.backbone_model}")
    logger.info(f"Head model: {args.head_model}")
    logger.info(f"PCA components: {args.n_components}")
    logger.info(f"Target column: {args.target_col}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("=" * 80)

    # Load data
    logger.info("\nLoading data...")
    Xs, Ys = load_data(args.input_dir, args.backbone_model, args.target_col)
    logger.info(f"Loaded {len(Xs)} samples with {Xs.shape[1]} features")

    # Perform 5-fold cross validation
    cv_scores, fold_results = perform_cross_validation(Xs, Ys, args)

    # Calculate and display cross validation statistics
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)

    logger.info("=" * 60)
    logger.info("5-Fold Cross Validation Results")
    logger.info(f"Mean CV Spearman: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
    logger.info(f"Min CV Spearman: {np.min(cv_scores):.4f}")
    logger.info(f"Max CV Spearman: {np.max(cv_scores):.4f}")
    logger.info(f"Individual fold scores: {[f'{s:.4f}' for s in cv_scores]}")

    # Save cross validation results
    df_cv_results = pd.DataFrame(fold_results)
    cv_results_path = os.path.join(args.output_dir, f'{args.backbone_model}_{args.head_model}_cv_results.csv')
    df_cv_results.to_csv(cv_results_path, index=False)
    logger.info(f"CV results saved to {cv_results_path}")

    # Save summary statistics
    summary = {
        'backbone_model': args.backbone_model,
        'head_model': args.head_model,
        'n_components': args.n_components,
        'seed': args.seed,
        'n_samples': len(Xs),
        'n_features': Xs.shape[1],
        'mean_cv_spearman': mean_cv_score,
        'std_cv_spearman': std_cv_score,
        'min_cv_spearman': np.min(cv_scores),
        'max_cv_spearman': np.max(cv_scores)
    }

    # Train final model on all data unless skipped
    if not args.no_final_model:
        final_model, pca_model, train_spearman = train_final_model(Xs, Ys, args)
        summary['final_train_spearman'] = train_spearman
        logger.info("=" * 60)
        logger.info("Final Model Training Complete")
        logger.info(f"Final model training Spearman: {train_spearman:.4f}")

    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(args.output_dir, 'training_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\nTraining summary saved to {summary_path}")

    # Print final summary
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Cross-validation: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
    if not args.no_final_model:
        logger.info(f"Final model training: {train_spearman:.4f}")
    logger.info(f"All results saved to: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
