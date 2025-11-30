#!/usr/bin/env python3
"""
ESM-IF Log-Likelihood Calculation Script

This script calculates log-likelihood (LLH) values for protein sequences using the ESM-IF
(Inverse Folding) model, which incorporates protein structure information. It scores sequences
based on how well they fit a given 3D structure.

Usage:
    python esm_if_llh.py -i <data.csv> -w <wt.fasta> -p <structure.pdb> [-c <chain>]

Example:
    python esm_if_llh.py -i data.csv -w wt.fasta -p wt_struct.pdb -c A
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import esm
import esm.inverse_folding
from Bio import SeqIO
from tqdm import tqdm
from copy import deepcopy
from loguru import logger
from scipy.stats import spearmanr, pearsonr


def load_esm_if_model(device='cuda'):
    """
    Load ESM-IF (Inverse Folding) model.

    Args:
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        model: ESM-IF model
        alphabet: ESM alphabet
    """
    logger.info("Loading ESM-IF model: esm_if1_gvp4_t16_142M_UR50")

    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model.eval()  # Disable dropout for inference

    # Move model to device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"Model loaded on device: {device}")

    return model, alphabet


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


def get_wt_native_seq_diff(wt_seq, native_seq):
    """
    Find alignment between wild-type sequence and native structure sequence.

    Args:
        wt_seq: Wild-type sequence from FASTA
        native_seq: Native sequence from PDB structure

    Returns:
        wt_start, wt_end: Indices in wild-type sequence
        native_start, native_end: Indices in native sequence
    """
    logger.warning(
        f"Wild-type sequence does not match native structure sequence.\n"
        f"  WT seq length: {len(wt_seq)}\n"
        f"  Native seq length: {len(native_seq)}"
    )

    # Find the longest common substring between wt_seq and native_seq
    max_substring = ''
    for i in range(len(wt_seq)):
        for j in range(i + 1, len(wt_seq) + 1):
            substring = wt_seq[i:j]
            if substring in native_seq and len(substring) > len(max_substring):
                max_substring = substring

    # Obtain the start and end indices
    wt_start = wt_seq.find(max_substring)
    wt_end = wt_start + len(max_substring)

    native_start = native_seq.find(max_substring)
    native_end = native_start + len(max_substring)

    logger.info(
        f"Aligned region:\n"
        f"  WT positions: {wt_start+1}-{wt_end} (length: {wt_end-wt_start})\n"
        f"  Native positions: {native_start+1}-{native_end} (length: {native_end-native_start})"
    )

    return wt_start, wt_end, native_start, native_end


def calculate_sequence_llh(seq, coords, wt_seq, native_seq, mutation_positions,
                          model, alphabet, masked=False):
    """
    Calculate log-likelihood for a single sequence.

    Args:
        seq: Sequence to score
        coords: Structure coordinates
        wt_seq: Wild-type sequence
        native_seq: Native sequence from structure
        mutation_positions: List of mutation positions (1-indexed)
        model: ESM-IF model
        alphabet: ESM alphabet
        masked: Whether to mask mutation positions

    Returns:
        Log-likelihood relative to wild-type
    """
    # Handle sequence alignment if wt and native differ
    if wt_seq != native_seq:
        wt_start, wt_end, native_start, native_end = get_wt_native_seq_diff(wt_seq, native_seq)
        # Map variant sequence to native structure sequence
        seq_to_score = native_seq[:native_start] + seq[wt_start:wt_end] + native_seq[native_end:]
    else:
        seq_to_score = seq

    # Optionally mask mutation positions
    masked_coords = deepcopy(coords)
    if masked and mutation_positions:
        for pos in mutation_positions:
            if 0 <= pos - 1 < len(masked_coords):
                masked_coords[pos - 1] = np.inf

    # Score the sequence
    ll_seq, _ = esm.inverse_folding.util.score_sequence(
        model, alphabet, masked_coords, str(seq_to_score)
    )

    return ll_seq


def calculate_esm_if_llh(data_csv, wt_fasta, pdb_file, chain='A',
                        masked=False, device='cuda', output_col='esmif_llh'):
    """
    Calculate ESM-IF log-likelihood for sequences in a CSV file.

    Args:
        data_csv: Path to CSV file with 'seq' column
        wt_fasta: Path to wild-type FASTA file
        pdb_file: Path to PDB structure file
        chain: Chain ID to use from PDB
        masked: Whether to mask mutation positions
        device: Device to use ('cuda' or 'cpu')
        output_col: Name for output column

    Returns:
        DataFrame with original data plus ESM-IF LLH column
    """
    data_csv = Path(data_csv)
    wt_fasta = Path(wt_fasta)
    pdb_file = Path(pdb_file)

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

    # Load structure
    logger.info(f"Loading structure from {pdb_file}, chain {chain}")
    if not pdb_file.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")

    coords, native_seq = esm.inverse_folding.util.load_coords(str(pdb_file), chain)
    logger.info(f"Native sequence length from structure: {len(native_seq)}")

    # Check if wild-type matches native
    if wt_seq != native_seq:
        logger.warning("Wild-type sequence differs from native structure sequence")
        wt_start, wt_end, native_start, native_end = get_wt_native_seq_diff(wt_seq, native_seq)
    else:
        logger.info("Wild-type sequence matches native structure sequence")
        wt_start, wt_end = 0, len(wt_seq)
        native_start, native_end = 0, len(native_seq)

    # Load ESM-IF model
    model, alphabet = load_esm_if_model(device)

    # Calculate wild-type log-likelihood
    logger.info("Calculating wild-type sequence log-likelihood...")
    ll_wt, _ = esm.inverse_folding.util.score_sequence(model, alphabet, coords, native_seq)
    logger.info(f"Wild-type log-likelihood: {ll_wt:.4f}")

    # Load sequence data
    logger.info(f"Loading sequence data from {data_csv}")
    df_data = pd.read_csv(data_csv)
    logger.info(f"Loaded {len(df_data)} variants")

    if 'seq' not in df_data.columns:
        raise ValueError(
            f"'seq' column not found in CSV. "
            f"Available columns: {df_data.columns.tolist()}"
        )

    # Derive mutations from sequences
    logger.info("Deriving mutations from 'seq' column...")
    mutations_list = []
    for seq in df_data['seq']:
        if len(seq) != len(wt_seq):
            logger.warning(f"Sequence length mismatch: {len(seq)} vs {len(wt_seq)} (wt)")
            mutations_list.append([])
            continue

        # Find mutation positions
        positions = []
        for i, (wt_aa, mut_aa) in enumerate(zip(wt_seq, seq)):
            if wt_aa != mut_aa:
                positions.append(i + 1)  # 1-indexed
        mutations_list.append(positions)

    # Calculate log-likelihoods
    logger.info(f"Calculating ESM-IF log-likelihoods (masked={masked})...")
    llhs = []

    for seq, mutation_positions in tqdm(
        zip(df_data['seq'], mutations_list),
        total=len(df_data),
        desc="Computing ESM-IF LLH",
        ncols=80
    ):
        if not mutation_positions:  # Wild-type
            llhs.append(0.0)
            continue

        try:
            ll_seq = calculate_sequence_llh(
                seq, coords, wt_seq, native_seq, mutation_positions,
                model, alphabet, masked
            )
            llhs.append(ll_seq - ll_wt)
        except Exception as e:
            logger.warning(f"Failed to calculate LLH for sequence: {e}")
            llhs.append(None)

    # Add to dataframe
    df_results = df_data.copy()
    df_results[output_col] = llhs

    # Report statistics
    valid_llhs = np.array([x for x in llhs if x is not None and not pd.isna(x)])
    logger.info("\nESM-IF Log-Likelihood Statistics:")
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
        description='Calculate ESM-IF log-likelihood for protein sequences using structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate ESM-IF LLH with structure
  python esm_if_llh.py -i data.csv -w wt.fasta -p structure.pdb

  # Specify chain and mask mutations
  python esm_if_llh.py -i data.csv -w wt.fasta -p structure.pdb -c A --masked

  # Custom output file
  python esm_if_llh.py -i data.csv -w wt.fasta -p structure.pdb -o results.csv

  # Use CPU instead of GPU
  python esm_if_llh.py -i data.csv -w wt.fasta -p structure.pdb --device cpu

Input Format:
  The input CSV must contain a 'seq' column with protein sequences.
  Mutations are automatically derived by comparing each sequence to the wild-type.

Structure Requirements:
  - PDB file with 3D coordinates
  - Chain ID (default: A)
  - Structure sequence should match or overlap with wild-type sequence
        """
    )

    parser.add_argument(
        '-i', '--input_csv',
        type=str,
        required=True,
        help='Path to CSV file containing sequences'
    )
    parser.add_argument(
        '-w', '--wt_fasta',
        type=str,
        required=True,
        help='Path to wild-type FASTA file'
    )
    parser.add_argument(
        '-p', '--pdb',
        type=str,
        required=True,
        help='Path to PDB structure file'
    )
    parser.add_argument(
        '-c', '--chain',
        type=str,
        default='A',
        help='Chain ID from PDB file (default: A)'
    )
    parser.add_argument(
        '--masked',
        action='store_true',
        default=False,
        help='Mask mutation positions in structure (default: False)'
    )
    parser.add_argument(
        '-o', '--output_csv',
        type=str,
        default=None,
        help='Output CSV file path. If not provided, uses <input_csv>_esmif_llh.csv'
    )
    parser.add_argument(
        '--output_col',
        type=str,
        default='esmif_llh',
        help='Name for output LLH column (default: esmif_llh)'
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
        help='Column name containing fitness values for correlation evaluation'
    )

    return parser.parse_args()


def main():
    """Main pipeline."""
    args = get_args()

    # Setup logging
    logger.info("=" * 80)
    logger.info("ESM-IF Log-Likelihood Calculation")
    logger.info("=" * 80)
    logger.info(f"Input CSV: {args.input_csv}")
    logger.info(f"Wild-type FASTA: {args.wt_fasta}")
    logger.info(f"PDB structure: {args.pdb}")
    logger.info(f"Chain: {args.chain}")
    logger.info(f"Masked: {args.masked}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 80)

    try:
        # Calculate log-likelihoods
        df_results = calculate_esm_if_llh(
            data_csv=args.input_csv,
            wt_fasta=args.wt_fasta,
            pdb_file=args.pdb,
            chain=args.chain,
            masked=args.masked,
            device=args.device,
            output_col=args.output_col
        )

        # Evaluate correlation with fitness if available
        fitness_col_found = None
        if args.fitness_col:
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

            llh_series = pd.to_numeric(df_results[args.output_col], errors='coerce')
            fitness_series = pd.to_numeric(df_results[fitness_col_found], errors='coerce')

            valid_mask = ~(pd.isna(llh_series) | pd.isna(fitness_series))
            llh_values = llh_series[valid_mask].values
            fitness_values = fitness_series[valid_mask].values

            if len(llh_values) > 2:
                spearman_r, spearman_p = spearmanr(llh_values, fitness_values)
                pearson_r, pearson_p = pearsonr(llh_values, fitness_values)

                logger.info("=" * 80)
                logger.info("Correlation with Fitness")
                logger.info("=" * 80)
                logger.info(f"  Valid pairs: {len(llh_values)} / {len(df_results)}")
                logger.info(f"  Spearman correlation: {spearman_r:.4f} (p={spearman_p:.2e})")
                logger.info(f"  Pearson correlation:  {pearson_r:.4f} (p={pearson_p:.2e})")
                logger.info("=" * 80)
            else:
                logger.warning(f"Not enough valid pairs for correlation: {len(llh_values)}")

        # Determine output path
        if args.output_csv:
            output_path = args.output_csv
        else:
            input_path = Path(args.input_csv)
            output_path = str(input_path.parent / f"{input_path.name}_esmif_llh.csv")

        # Save results
        df_results.to_csv(output_path, index=False)

        logger.info("\n" + "=" * 80)
        logger.info("Calculation Complete!")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"Total variants processed: {len(df_results)}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error("=" * 80)
        logger.error("Calculation Failed!")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}")
        logger.error("=" * 80)
        raise


if __name__ == "__main__":
    exit(main())
