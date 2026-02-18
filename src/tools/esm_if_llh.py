"""
ESM-IF log-likelihood calculation tools for structure-based protein fitness prediction.

This MCP Server provides 1 tool:
1. esm_if_calculate_llh: Calculate ESM-IF log-likelihood using protein structure

The tool uses the ESM-IF (Inverse Folding) model to score protein sequences
based on how well they fit a given 3D structure.
"""

# Standard imports
from typing import Annotated, Literal, Optional, TYPE_CHECKING
import pandas as pd
import numpy as np
import torch
import esm
import esm.inverse_folding
from Bio import SeqIO
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
import os
from fastmcp import FastMCP
from datetime import datetime
from scipy.stats import spearmanr, pearsonr

if TYPE_CHECKING:
    from job_queue import QueueManager

# MCP server instance
esm_if_llh_mcp = FastMCP(name="esm_if_llh")


def load_esm_if_model(device='cuda'):
    """Load ESM-IF (Inverse Folding) model.

    Args:
        device: Device to use (e.g., 'cuda', 'cuda:0', 'cuda:1', 'cpu')
    """
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model.eval()  # Disable dropout for inference

    # Move model to device
    # Support formats: 'cuda', 'cuda:0', 'cuda:1', 'cpu'
    if device.startswith('cuda') and torch.cuda.is_available():
        device_obj = torch.device(device)
    else:
        device_obj = torch.device('cpu')
    model = model.to(device_obj)

    return model, alphabet


def read_fasta(fasta_path):
    """Read FASTA file and return sequences."""
    sequences = [str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")]
    descriptions = [record.description for record in SeqIO.parse(fasta_path, "fasta")]
    return sequences, descriptions


def get_wt_native_seq_diff(wt_seq, native_seq):
    """Find alignment between wild-type sequence and native structure sequence."""
    # Find the longest common substring
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

    return wt_start, wt_end, native_start, native_end


def calculate_sequence_llh(seq, coords, wt_seq, native_seq, mutation_positions,
                          model, alphabet, masked=False):
    """Calculate log-likelihood for a single sequence."""
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


@esm_if_llh_mcp.tool
def esm_if_calculate_llh(
    data_csv: Annotated[str, "Path to CSV file containing sequences"],
    wt_fasta: Annotated[str, "Path to wild-type FASTA file"],
    pdb_file: Annotated[str, "Path to PDB structure file"],
    chain: Annotated[str, "Chain ID to use from PDB"] = 'A',
    masked: Annotated[bool, "Whether to mask mutation positions in structure"] = False,
    device: Annotated[str, "Device to use (e.g., 'cuda', 'cuda:0', 'cuda:1', 'cpu')"] = "cuda",
    output_col: Annotated[str, "Name for output column"] = "esmif_llh",
    output_csv: Annotated[str | None, "Output CSV file path. If None, uses <input_csv>_esmif_llh.csv"] = None,
    fitness_col: Annotated[str | None, "Column name containing fitness values for correlation evaluation"] = None,
) -> dict:
    """
    Calculate ESM-IF log-likelihood for protein sequences using structure.

    This tool:
    1. Reads CSV file with 'seq' column, wild-type FASTA, and PDB structure
    2. Loads ESM-IF model (Inverse Folding)
    3. Derives mutations by comparing sequences to wild-type
    4. Calculates structure-based log-likelihood for each variant
    5. Returns results with optional correlation statistics

    Input: CSV with sequences, wild-type FASTA, PDB structure, parameters
    Output: Dictionary with results path, statistics, and correlation metrics
    """
    try:
        data_csv = Path(data_csv)
        wt_fasta = Path(wt_fasta)
        pdb_file = Path(pdb_file)

        # Load wild-type sequence
        if not wt_fasta.exists():
            raise FileNotFoundError(f"Wild-type FASTA file not found: {wt_fasta}")

        wt_seqs, wt_descriptions = read_fasta(wt_fasta)
        if len(wt_seqs) == 0:
            raise ValueError(f"No sequences found in {wt_fasta}")

        wt_seq = wt_seqs[0]

        # Load structure
        if not pdb_file.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")

        coords, native_seq = esm.inverse_folding.util.load_coords(str(pdb_file), chain)

        # Check if wild-type matches native
        if wt_seq != native_seq:
            wt_start, wt_end, native_start, native_end = get_wt_native_seq_diff(wt_seq, native_seq)
            alignment_info = {
                "wt_region": f"{wt_start+1}-{wt_end}",
                "native_region": f"{native_start+1}-{native_end}",
                "aligned_length": wt_end - wt_start,
            }
        else:
            wt_start, wt_end = 0, len(wt_seq)
            native_start, native_end = 0, len(native_seq)
            alignment_info = {
                "status": "perfect_match",
            }

        # Load ESM-IF model
        model, alphabet = load_esm_if_model(device)

        # Calculate wild-type log-likelihood
        ll_wt, _ = esm.inverse_folding.util.score_sequence(model, alphabet, coords, native_seq)

        # Load sequence data
        df_data = pd.read_csv(data_csv)

        if 'seq' not in df_data.columns:
            raise ValueError(
                f"'seq' column not found in CSV. "
                f"Available columns: {df_data.columns.tolist()}"
            )

        # Derive mutations from sequences
        mutations_list = []
        for seq in df_data['seq']:
            if len(seq) != len(wt_seq):
                mutations_list.append([])
                continue

            # Find mutation positions
            positions = []
            for i, (wt_aa, mut_aa) in enumerate(zip(wt_seq, seq)):
                if wt_aa != mut_aa:
                    positions.append(i + 1)  # 1-indexed
            mutations_list.append(positions)

        # Calculate log-likelihoods
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
                llhs.append(None)

        # Add to dataframe
        df_results = df_data.copy()
        df_results[output_col] = llhs

        # Report statistics
        valid_llhs = np.array([x for x in llhs if x is not None and not pd.isna(x)])
        stats = {
            "valid_llh_count": int(len(valid_llhs)),
            "total_count": int(len(llhs)),
        }

        if len(valid_llhs) > 0:
            stats.update({
                "mean": float(np.mean(valid_llhs)),
                "std": float(np.std(valid_llhs)),
                "min": float(np.min(valid_llhs)),
                "max": float(np.max(valid_llhs)),
            })

        # Evaluate correlation with fitness if available
        correlation_results = None
        fitness_col_found = None

        if fitness_col:
            if fitness_col in df_results.columns:
                fitness_col_found = fitness_col
        else:
            # Auto-detect fitness column
            for col in ['fitness', 'log_fitness']:
                if col in df_results.columns:
                    fitness_col_found = col
                    break

        if fitness_col_found:
            llh_series = pd.to_numeric(df_results[output_col], errors='coerce')
            fitness_series = pd.to_numeric(df_results[fitness_col_found], errors='coerce')

            valid_mask = ~(pd.isna(llh_series) | pd.isna(fitness_series))
            llh_values = llh_series[valid_mask].values
            fitness_values = fitness_series[valid_mask].values

            if len(llh_values) > 2:
                spearman_r, spearman_p = spearmanr(llh_values, fitness_values)
                pearson_r, pearson_p = pearsonr(llh_values, fitness_values)

                correlation_results = {
                    "fitness_column": fitness_col_found,
                    "valid_pairs": int(len(llh_values)),
                    "spearman_r": float(spearman_r),
                    "spearman_p": float(spearman_p),
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                }

        # Determine output path
        if output_csv:
            output_path = output_csv
        else:
            output_path = str(data_csv.parent / f"{data_csv.name}_esmif_llh.csv")

        # Save results
        df_results.to_csv(output_path, index=False)

        return {
            "status": "success",
            "output_csv": output_path,
            "model_name": "esm_if1_gvp4_t16_142M_UR50",
            "wt_sequence_length": len(wt_seq),
            "native_sequence_length": len(native_seq),
            "chain": chain,
            "masked": masked,
            "wt_llh": float(ll_wt),
            "alignment": alignment_info,
            "total_variants": len(df_results),
            "statistics": stats,
            "correlation": correlation_results,
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "data_csv": str(data_csv),
            "wt_fasta": str(wt_fasta),
            "pdb_file": str(pdb_file),
        }


# Implementation function for queue-based execution
def _esm_if_calculate_llh_impl(
    data_csv: str,
    wt_fasta: str,
    pdb_file: str,
    chain: str = 'A',
    masked: bool = False,
    device: str = "cuda",
    output_col: str = "esmif_llh",
    output_csv: Optional[str] = None,
    fitness_col: Optional[str] = None,
) -> dict:
    """Internal implementation for queue-based execution."""
    return esm_if_calculate_llh(
        data_csv=data_csv,
        wt_fasta=wt_fasta,
        pdb_file=pdb_file,
        chain=chain,
        masked=masked,
        device=device,
        output_col=output_col,
        output_csv=output_csv,
        fitness_col=fitness_col,
    )


def create_esm_if_llh_mcp(queue_manager: "QueueManager") -> FastMCP:
    """Create MCP instance with queue-wrapped tool.

    Args:
        queue_manager: Queue manager for job submission

    Returns:
        FastMCP instance with queue-wrapped esm_if_calculate_llh tool
    """
    from job_queue.job import Job

    queued_esm_if_llh_mcp = FastMCP(name="esm_if_llh")

    @queued_esm_if_llh_mcp.tool
    async def esm_if_calculate_llh(
        data_csv: Annotated[str, "Path to CSV file containing sequences"],
        wt_fasta: Annotated[str, "Path to wild-type FASTA file"],
        pdb_file: Annotated[str, "Path to PDB structure file"],
        chain: Annotated[str, "Chain ID to use from PDB"] = 'A',
        masked: Annotated[bool, "Whether to mask mutation positions in structure"] = False,
        device: Annotated[str, "Device (ignored - auto-assigned by queue)"] = "cuda",
        output_col: Annotated[str, "Name for output column"] = "esmif_llh",
        output_csv: Annotated[str | None, "Output CSV file path"] = None,
        fitness_col: Annotated[str | None, "Column name containing fitness values"] = None,
    ) -> dict:
        """
        Calculate ESM-IF log-likelihood for protein sequences (queue-managed).

        Jobs are queued and executed in FIFO order with automatic GPU assignment.
        """
        job = Job(
            tool_name="esm_if_calculate_llh",
            tool_module="tools.esm_if_llh",
            tool_function="_esm_if_calculate_llh_impl",
            kwargs={
                "data_csv": data_csv,
                "wt_fasta": wt_fasta,
                "pdb_file": pdb_file,
                "chain": chain,
                "masked": masked,
                "output_col": output_col,
                "output_csv": output_csv,
                "fitness_col": fitness_col,
                # device is injected by worker
            },
            requires_gpu=True,
        )
        return await queue_manager.submit(job)

    return queued_esm_if_llh_mcp
