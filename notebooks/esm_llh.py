#!/usr/bin/env python3
"""
ESM Log-Likelihood Calculation Script

This script calculates log-likelihood (LLH) values for protein mutations with respect to
a wild-type sequence using ESM models. It computes mutation probabilities at each position
and calculates the LLH ratio for variants.

Usage:
    python esm_llh.py -i <data.csv> -w <wt.fasta> -m <model_name>

Example:
    python esm_llh.py -i data.csv -w wt.fasta -m esm2_t33_650M_UR50D
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import esm
from Bio import SeqIO
from tqdm import tqdm
import multiprocessing as mp
from loguru import logger
from scipy.stats import spearmanr, pearsonr


# Natural amino acids
NATURAL_AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
              'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


def load_esm_model(model_name='esm2_t33_650M_UR50D', device='cuda'):
    """
    Load ESM model and components.

    Args:
        model_name: Name of the ESM model to load
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        model: ESM model
        alphabet: ESM alphabet
        batch_converter: Batch converter for processing sequences
    """
    logger.info(f"Loading ESM model: {model_name}")

    if model_name == 'esm2_t33_650M_UR50D':
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    elif model_name == 'esm2_t36_3B_UR50D':
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    elif model_name == 'esm2_t48_15B_UR50D':
        model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
    elif model_name == 'esm1v_t33_650M_UR90S_1':
        model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
    elif model_name == 'esm1v_t33_650M_UR90S_2':
        model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_2()
    elif model_name == 'esm1v_t33_650M_UR90S_3':
        model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_3()
    elif model_name == 'esm1v_t33_650M_UR90S_4':
        model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_4()
    elif model_name == 'esm1v_t33_650M_UR90S_5':
        model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_5()
    else:
        logger.warning(
            f"Model {model_name} not supported. Using 'esm2_t33_650M_UR50D' instead. "
            f"Supported models: esm2_t33_650M_UR50D, esm2_t36_3B_UR50D, esm2_t48_15B_UR50D, "
            f"esm1v_t33_650M_UR90S_[1-5]"
        )
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    batch_converter = alphabet.get_batch_converter()
    model.eval()  # Disable dropout for inference

    # Move model to device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"Model loaded on device: {device}")

    return model, alphabet, batch_converter


def get_pos_mutation_probabilities(sequence, position, model, alphabet, batch_converter, device='cuda'):
    """
    Get mutation probabilities for a specific position in the protein sequence.

    Args:
        sequence: Protein sequence
        position: Position of the residue (0-based indexing)
        model: ESM model
        alphabet: ESM alphabet
        batch_converter: ESM batch converter
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        Dictionary of {amino_acid: probability} for the position
    """
    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        token_logits = model(batch_tokens)["logits"]
        token_probs = torch.log_softmax(token_logits, dim=-1)

    # Position + 1 because of BOS token
    probabilities = torch.exp(token_probs[0, position + 1, :]).cpu().tolist()
    residues = [alphabet.get_tok(i) for i in range(len(probabilities))]
    position_probabilities = dict(zip(residues, probabilities))

    # Sort by probability (descending)
    position_probabilities = dict(
        sorted(position_probabilities.items(), key=lambda item: item[1], reverse=True)
    )

    return position_probabilities


def read_fasta(fasta_path):
    """
    Read FASTA file and return sequences.

    Args:
        fasta_path: Path to FASTA file

    Returns:
        sequences: List of sequences
        descriptions: List of descriptions
    """
    sequences = [str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")]
    descriptions = [record.description for record in SeqIO.parse(fasta_path, "fasta")]
    return sequences, descriptions


def get_mutation_probabilities_df(wt_seq, model, alphabet, batch_converter, device='cuda'):
    """
    Calculate mutation probabilities for all positions in the wild-type sequence.

    Args:
        wt_seq: Wild-type sequence
        model: ESM model
        alphabet: ESM alphabet
        batch_converter: Batch converter
        device: Device to run on

    Returns:
        DataFrame with columns: pos, ref, A, C, D, E, ..., Y
    """
    logger.info(f"Calculating mutation probabilities for sequence of length {len(wt_seq)}")

    site_probabilities = {}

    # Get mutation probabilities for each position
    for pos in tqdm(range(len(wt_seq)), desc="Computing probabilities", ncols=80):
        probabilities = get_pos_mutation_probabilities(
            wt_seq, pos, model, alphabet, batch_converter, device
        )
        site_probabilities[pos + 1] = probabilities  # 1-based indexing

    # Build dataframe
    df_mut_probs = pd.DataFrame({
        'pos': list(range(1, len(wt_seq) + 1)),
        'ref': list(wt_seq)
    })

    # Add probability columns for each amino acid
    for aa in NATURAL_AA:
        df_mut_probs[aa] = [
            site_probabilities[site].get(aa, 0) for site in site_probabilities.keys()
        ]

    logger.info(f"Mutation probabilities calculated for {len(df_mut_probs)} positions")

    return df_mut_probs


def calculate_loglikelihood(mutation, df_mut_probs):
    """
    Calculate log-likelihood for mutations with respect to wild-type.

    Args:
        mutation: Mutation string (e.g., "V39Y" or "V39Y/V54S" for multiple mutations)
        df_mut_probs: DataFrame with mutation probabilities

    Returns:
        Log-likelihood value
    """
    if not isinstance(mutation, str) or mutation == '' or pd.isna(mutation):
        return 0.0

    mutations = mutation.split('/')
    llh = 0.0

    for mut in mutations:
        # Parse mutation: ref_AA + position + mut_AA (e.g., "V39Y")
        if len(mut) < 3:
            logger.warning(f"Invalid mutation format: {mut}")
            return None

        ref_AA = mut[0]
        mut_AA = mut[-1]
        try:
            pos = int(mut[1:-1])
        except ValueError:
            logger.warning(f"Invalid position in mutation: {mut}")
            return None

        # Check if amino acids are in columns
        if ref_AA not in df_mut_probs.columns or mut_AA not in df_mut_probs.columns:
            logger.warning(f"Amino acid {ref_AA} or {mut_AA} not in mutation probabilities")
            return None

        # Check if position exists
        if pos not in df_mut_probs['pos'].values:
            logger.warning(f"Position {pos} not in mutation probabilities")
            return None

        # Get probabilities
        prob_ref = df_mut_probs.loc[df_mut_probs['pos'] == pos, ref_AA].values[0]
        prob_mut = df_mut_probs.loc[df_mut_probs['pos'] == pos, mut_AA].values[0]

        # Calculate log-likelihood ratio
        if prob_ref > 0:
            llh += np.log(prob_mut / prob_ref)
        else:
            logger.warning(f"Zero probability for reference AA {ref_AA} at position {pos}")
            return None

    return llh


def calculate_llh_wrapper(args):
    """Wrapper for multiprocessing."""
    mutation, df_mut_probs = args
    return calculate_loglikelihood(mutation, df_mut_probs)


def calculate_llh_batch(mutations, df_mut_probs, n_proc=None):
    """
    Calculate log-likelihoods for a batch of mutations using multiprocessing.

    Args:
        mutations: List of mutation strings
        df_mut_probs: DataFrame with mutation probabilities
        n_proc: Number of processes (None = auto)

    Returns:
        Array of log-likelihood values
    """
    if n_proc is None:
        n_proc = min(mp.cpu_count(), len(mutations))

    logger.info(f"Calculating LLH for {len(mutations)} mutations using {n_proc} processes")

    if n_proc == 1:
        # Single process mode
        llhs = [calculate_loglikelihood(mut, df_mut_probs) for mut in tqdm(mutations, ncols=80)]
    else:
        # Multiprocessing mode
        with mp.Pool(n_proc) as pool:
            args_for_llh = zip(mutations, [df_mut_probs] * len(mutations))
            llhs = list(
                tqdm(pool.imap(calculate_llh_wrapper, args_for_llh),
                     total=len(mutations),
                     ncols=80,
                     desc="Computing LLH")
            )

    return np.array(llhs)


def calculate_esm_llh(data_csv, wt_fasta, model_name='esm2_t33_650M_UR50D',
                     n_proc=None, device='cuda',
                     output_col='esm_llh'):
    """
    Calculate ESM log-likelihood for mutations in a CSV file.

    Args:
        data_csv: Path to CSV file with 'seq' column containing protein sequences
        wt_fasta: Path to wild-type FASTA file
        model_name: ESM model name
        n_proc: Number of processes for parallel computation
        device: Device to use ('cuda' or 'cpu')
        output_col: Name for output column

    Returns:
        DataFrame with original data plus LLH column
    """
    data_csv = Path(data_csv)
    wt_fasta = Path(wt_fasta)
    data_dir = data_csv.parent

    # Load wild-type sequence
    logger.info(f"Loading wild-type sequence from {wt_fasta}")
    if not wt_fasta.exists():
        raise FileNotFoundError(f"Wild-type FASTA file not found: {wt_fasta}")

    wt_seqs, wt_descriptions = read_fasta(wt_fasta)
    if len(wt_seqs) == 0:
        raise ValueError(f"No sequences found in {wt_fasta}")

    wt_seq = wt_seqs[0]
    logger.info(f"Wild-type sequence length: {len(wt_seq)}")
    logger.info(f"Wild-type description: {wt_descriptions[0]}")

    # Load sequence data
    logger.info(f"Loading sequence data from {data_csv}")
    df_data = pd.read_csv(data_csv)
    logger.info(f"Loaded {len(df_data)} variants")

    # Derive mutations from 'seq' column by comparing to wild-type
    if 'seq' not in df_data.columns:
        raise ValueError(
            f"'seq' column not found in CSV. "
            f"Available columns: {df_data.columns.tolist()}"
        )

    logger.info("Deriving mutations from 'seq' column by comparing to wild-type...")
    mutations = []
    for seq in df_data['seq']:
        if len(seq) != len(wt_seq):
            logger.warning(f"Sequence length mismatch: {len(seq)} vs {len(wt_seq)} (wt)")
            mutations.append(None)
            continue
        # Find all mutations by comparing to wild-type
        muts = []
        for i, (wt_aa, mut_aa) in enumerate(zip(wt_seq, seq)):
            if wt_aa != mut_aa:
                # Position is 1-indexed
                muts.append(f"{wt_aa}{i+1}{mut_aa}")
        mutations.append('/'.join(muts) if muts else '')
    df_data['_derived_mutations'] = mutations
    logger.info(f"Derived mutations for {len(mutations)} sequences")

    # Check for cached mutation probabilities
    mut_probs_file = data_dir / f"{wt_fasta.stem}_mut_probs_{model_name}.csv"

    recalculate = False
    if mut_probs_file.exists():
        logger.info(f"Loading cached mutation probabilities from {mut_probs_file}")
        df_mut_probs = pd.read_csv(mut_probs_file)
        
        # Validate cached data matches current wild-type sequence
        if len(df_mut_probs) != len(wt_seq):
            logger.warning(
                f"Cached mutation probabilities length ({len(df_mut_probs)}) "
                f"does not match wild-type sequence length ({len(wt_seq)}). "
                f"Recalculating..."
            )
            recalculate = True
        else:
            # Verify reference amino acids match
            cached_refs = ''.join(df_mut_probs['ref'].tolist())
            if cached_refs != wt_seq:
                logger.warning(
                    f"Cached reference sequence does not match wild-type sequence. "
                    f"Recalculating..."
                )
                recalculate = True
            else:
                logger.info(f"Cached mutation probabilities validated: {len(df_mut_probs)} positions")
    else:
        recalculate = True

    if recalculate:
        # Calculate mutation probabilities
        logger.info("Calculating mutation probabilities (this may take a while)...")
        model, alphabet, batch_converter = load_esm_model(model_name, device)
        df_mut_probs = get_mutation_probabilities_df(
            wt_seq, model, alphabet, batch_converter, device
        )

        # Save mutation probabilities for future use
        df_mut_probs.to_csv(mut_probs_file, index=False)
        logger.info(f"Mutation probabilities saved to {mut_probs_file}")

    # Calculate log-likelihoods
    logger.info("Calculating log-likelihoods for variants...")
    llhs = calculate_llh_batch(df_data['_derived_mutations'].tolist(), df_mut_probs, n_proc)

    # Add to dataframe
    df_results = df_data.copy()
    df_results[output_col] = llhs

    # Report statistics
    valid_llhs = llhs[~pd.isna(llhs)]
    logger.info("\nLog-Likelihood Statistics:")
    logger.info(f"  Valid LLH values: {len(valid_llhs)} / {len(llhs)}")
    if len(valid_llhs) > 0:
        logger.info(f"  Mean: {np.mean(valid_llhs):.4f}")
        logger.info(f"  Std:  {np.std(valid_llhs):.4f}")
        logger.info(f"  Min:  {np.min(valid_llhs):.4f}")
        logger.info(f"  Max:  {np.max(valid_llhs):.4f}")

    return df_results


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate ESM log-likelihood for protein mutations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate LLH using default model
  python esm_llh.py -i data.csv -w wt.fasta

  # Use larger ESM model
  python esm_llh.py -i data.csv -w wt.fasta -m esm2_t36_3B_UR50D

  # Custom output file
  python esm_llh.py -i data.csv -w wt.fasta -o results.csv

  # Use CPU instead of GPU
  python esm_llh.py -i data.csv -w wt.fasta --device cpu

Input Format:
  The input CSV must contain a 'seq' column with protein sequences.
  Mutations are automatically derived by comparing each sequence to the wild-type.

Available Models:
  - esm2_t33_650M_UR50D (default)
  - esm2_t36_3B_UR50D
  - esm2_t48_15B_UR50D
  - esm1v_t33_650M_UR90S_1
  - esm1v_t33_650M_UR90S_2
  - esm1v_t33_650M_UR90S_3
  - esm1v_t33_650M_UR90S_4
  - esm1v_t33_650M_UR90S_5
        """
    )

    parser.add_argument(
        '-i', '--input_csv',
        type=str,
        required=True,
        help='Path to CSV file containing mutations'
    )
    parser.add_argument(
        '-w', '--wt_fasta',
        type=str,
        required=True,
        help='Path to wild-type FASTA file'
    )
    parser.add_argument(
        '-m', '--model_name',
        type=str,
        default='esm2_t33_650M_UR50D',
        choices=[
            'esm2_t33_650M_UR50D',
            'esm2_t36_3B_UR50D',
            'esm2_t48_15B_UR50D',
            'esm1v_t33_650M_UR90S_1',
            'esm1v_t33_650M_UR90S_2',
            'esm1v_t33_650M_UR90S_3',
            'esm1v_t33_650M_UR90S_4',
            'esm1v_t33_650M_UR90S_5'
        ],
        help='ESM model to use (default: esm2_t33_650M_UR50D)'
    )
    parser.add_argument(
        '-o', '--output_csv',
        type=str,
        default=None,
        help='Output CSV file path. If not provided, uses <input_csv>_esm_llh.csv'
    )
    parser.add_argument(
        '--output_col',
        type=str,
        default='esm_llh',
        help='Name for output LLH column (default: esm_llh)'
    )
    parser.add_argument(
        '--n_proc',
        type=int,
        default=None,
        help='Number of processes for parallel LLH calculation (default: auto)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    parser.add_argument(
        '--fitness_col',
        type=str,
        default=None,
        help='Column name containing fitness values for correlation evaluation. '
             'If not specified, will auto-detect "fitness" or "log_fitness" columns.'
    )

    return parser.parse_args()


def main():
    """Main pipeline."""
    args = get_args()

    # Setup logging
    logger.info("=" * 60)
    logger.info("ESM Log-Likelihood Calculation")
    logger.info(f"Input CSV: {args.input_csv}")
    logger.info(f"Wild-type FASTA: {args.wt_fasta}")
    logger.info(f"ESM model: {args.model_name}")
    logger.info(f"Fitness column: {args.fitness_col if args.fitness_col else 'auto-detect'}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 60)

    try:
        # Calculate log-likelihoods
        df_results = calculate_esm_llh(
            data_csv=args.input_csv,
            wt_fasta=args.wt_fasta,
            model_name=args.model_name,
            n_proc=args.n_proc,
            device=args.device,
            output_col=args.output_col
        )

        # Evaluate correlation with fitness if available
        fitness_col_found = None
        if args.fitness_col:
            # User specified fitness column
            if args.fitness_col in df_results.columns:
                fitness_col_found = args.fitness_col
            else:
                logger.warning(f"Specified fitness column '{args.fitness_col}' not found in CSV")
        else:
            # Auto-detect fitness column
            for col in ['fitness', 'log_fitness']:
                if col in df_results.columns:
                    fitness_col_found = col
                    break

        if fitness_col_found:
            logger.info(f"\nEvaluating with fitness column: '{fitness_col_found}'")
            
            # Convert columns to numeric, coercing errors to NaN
            llh_series = pd.to_numeric(df_results[args.output_col], errors='coerce')
            fitness_series = pd.to_numeric(df_results[fitness_col_found], errors='coerce')
            
            # Get valid pairs (both LLH and fitness are not NaN and not None)
            valid_mask = ~(pd.isna(llh_series) | pd.isna(fitness_series))
            llh_values = llh_series[valid_mask].values
            fitness_values = fitness_series[valid_mask].values
            
            if len(llh_values) > 2:
                spearman_r, spearman_p = spearmanr(llh_values, fitness_values)
                pearson_r, pearson_p = pearsonr(llh_values, fitness_values)
                
                logger.info("=" * 60)
                logger.info("Correlation with Fitness")
                logger.info("=" * 60)
                logger.info(f"  Valid pairs: {len(llh_values)} / {len(df_results)}")
                logger.info(f"  Spearman correlation: {spearman_r:.4f} (p={spearman_p:.2e})")
                logger.info(f"  Pearson correlation:  {pearson_r:.4f} (p={pearson_p:.2e})")
                logger.info("=" * 60)
            else:
                logger.warning(f"Not enough valid pairs for correlation: {len(llh_values)}")
        else:
            logger.info("\nNo fitness column found for correlation evaluation. Skipping.")

        # Determine output path
        if args.output_csv:
            output_path = args.output_csv
        else:
            input_path = Path(args.input_csv)
            output_path = str(input_path.parent / f"{input_path.name}_esm_llh.csv")

        # Save results
        df_results.to_csv(output_path, index=False)

        logger.info("=" * 60)
        logger.info("Calculation Complete!")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"Total variants processed: {len(df_results)}")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error("=" * 60)
        logger.error("Calculation Failed!")
        logger.error("=" * 60)
        logger.error(f"Error: {str(e)}")
        logger.error("=" * 60)
        raise


if __name__ == "__main__":
    exit(main())
