"""
ESM log-likelihood calculation tools for protein fitness prediction.

This MCP Server provides 1 tool:
1. esm_calculate_llh: Calculate ESM log-likelihood for protein mutations

The tool reads a CSV file with a 'seq' column, compares to wild-type sequence,
and calculates log-likelihood values for mutations using ESM models.
"""

# Standard imports
from typing import Annotated, Literal, Optional, TYPE_CHECKING
import pandas as pd
import numpy as np
import torch
import esm
from Bio import SeqIO
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
import os
from fastmcp import FastMCP
from datetime import datetime
from scipy.stats import spearmanr, pearsonr

if TYPE_CHECKING:
    from job_queue import QueueManager

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Natural amino acids
NATURAL_AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
              'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# MCP server instance
esm_llh_mcp = FastMCP(name="esm_llh")


def load_esm_model(model_name='esm2_t33_650M_UR50D', device='cuda'):
    """Load ESM model and components.

    Args:
        model_name: Name of the ESM model to load
        device: Device to use (e.g., 'cuda', 'cuda:0', 'cuda:1', 'cpu')
    """
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
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    batch_converter = alphabet.get_batch_converter()
    model.eval()  # Disable dropout for inference

    # Move model to device
    # Support formats: 'cuda', 'cuda:0', 'cuda:1', 'cpu'
    if device.startswith('cuda') and torch.cuda.is_available():
        device_obj = torch.device(device)
    else:
        device_obj = torch.device('cpu')
    model = model.to(device_obj)

    return model, alphabet, batch_converter


def get_pos_mutation_probabilities(sequence, position, model, alphabet, batch_converter, device='cuda'):
    """Get mutation probabilities for a specific position.

    Args:
        device: Device to use (e.g., 'cuda', 'cuda:0', 'cuda:1', 'cpu')
    """
    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    if device.startswith('cuda') and torch.cuda.is_available():
        device_obj = torch.device(device)
    else:
        device_obj = torch.device('cpu')
    batch_tokens = batch_tokens.to(device_obj)

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
    """Read FASTA file and return sequences."""
    sequences = [str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")]
    descriptions = [record.description for record in SeqIO.parse(fasta_path, "fasta")]
    return sequences, descriptions


def get_mutation_probabilities_df(wt_seq, model, alphabet, batch_converter, device='cuda'):
    """Calculate mutation probabilities for all positions."""
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

    return df_mut_probs


def calculate_loglikelihood(mutation, df_mut_probs):
    """Calculate log-likelihood for mutations."""
    if not isinstance(mutation, str) or mutation == '' or pd.isna(mutation):
        return 0.0

    mutations = mutation.split('/')
    llh = 0.0

    for mut in mutations:
        if len(mut) < 3:
            return None

        ref_AA = mut[0]
        mut_AA = mut[-1]
        try:
            pos = int(mut[1:-1])
        except ValueError:
            return None

        if ref_AA not in df_mut_probs.columns or mut_AA not in df_mut_probs.columns:
            return None

        if pos not in df_mut_probs['pos'].values:
            return None

        # Get probabilities
        prob_ref = df_mut_probs.loc[df_mut_probs['pos'] == pos, ref_AA].values[0]
        prob_mut = df_mut_probs.loc[df_mut_probs['pos'] == pos, mut_AA].values[0]

        # Calculate log-likelihood ratio
        if prob_ref > 0:
            llh += np.log(prob_mut / prob_ref)
        else:
            return None

    return llh


def calculate_llh_wrapper(args):
    """Wrapper for multiprocessing."""
    mutation, df_mut_probs = args
    return calculate_loglikelihood(mutation, df_mut_probs)


def calculate_llh_batch(mutations, df_mut_probs, n_proc=None):
    """Calculate log-likelihoods for a batch of mutations."""
    if n_proc is None:
        n_proc = min(mp.cpu_count(), len(mutations))

    if n_proc == 1:
        llhs = [calculate_loglikelihood(mut, df_mut_probs) for mut in tqdm(mutations, ncols=80)]
    else:
        with mp.Pool(n_proc) as pool:
            args_for_llh = zip(mutations, [df_mut_probs] * len(mutations))
            llhs = list(
                tqdm(pool.imap(calculate_llh_wrapper, args_for_llh),
                     total=len(mutations),
                     ncols=80,
                     desc="Computing LLH")
            )

    return np.array(llhs)


@esm_llh_mcp.tool
def esm_calculate_llh(
    data_csv: Annotated[str, "Path to CSV file containing sequences"],
    wt_fasta: Annotated[str, "Path to wild-type FASTA file"],
    model_name: Annotated[
        Literal[
            "esm2_t33_650M_UR50D",
            "esm2_t36_3B_UR50D",
            "esm2_t48_15B_UR50D",
            "esm1v_t33_650M_UR90S_1",
            "esm1v_t33_650M_UR90S_2",
            "esm1v_t33_650M_UR90S_3",
            "esm1v_t33_650M_UR90S_4",
            "esm1v_t33_650M_UR90S_5"
        ],
        "ESM model name to use"
    ] = "esm2_t33_650M_UR50D",
    output_col: Annotated[str, "Name for output column"] = "esm_llh",
    output_csv: Annotated[str | None, "Output CSV file path. If None, uses <input_csv>_esm_llh.csv"] = None,
    n_proc: Annotated[int | None, "Number of processes for parallel computation"] = None,
    device: Annotated[str, "Device to use (e.g., 'cuda', 'cuda:0', 'cuda:1', 'cpu')"] = "cuda",
    fitness_col: Annotated[str | None, "Column name containing fitness values for correlation evaluation"] = None,
) -> dict:
    """
    Calculate ESM log-likelihood for protein mutations.

    This tool:
    1. Reads CSV file with 'seq' column and wild-type FASTA
    2. Derives mutations by comparing sequences to wild-type
    3. Calculates mutation probabilities using ESM model
    4. Computes log-likelihood for each variant
    5. Returns results with optional correlation statistics

    Input: CSV with sequences, wild-type FASTA, model parameters
    Output: Dictionary with results path, statistics, and correlation metrics
    """
    try:
        data_csv = Path(data_csv)
        wt_fasta = Path(wt_fasta)
        data_dir = data_csv.parent

        # Load wild-type sequence
        if not wt_fasta.exists():
            raise FileNotFoundError(f"Wild-type FASTA file not found: {wt_fasta}")

        wt_seqs, wt_descriptions = read_fasta(wt_fasta)
        if len(wt_seqs) == 0:
            raise ValueError(f"No sequences found in {wt_fasta}")

        wt_seq = wt_seqs[0]

        # Load sequence data
        df_data = pd.read_csv(data_csv)

        # Derive mutations from 'seq' column
        if 'seq' not in df_data.columns:
            raise ValueError(
                f"'seq' column not found in CSV. "
                f"Available columns: {df_data.columns.tolist()}"
            )

        mutations = []
        for seq in df_data['seq']:
            if len(seq) != len(wt_seq):
                mutations.append(None)
                continue
            # Find all mutations
            muts = []
            for i, (wt_aa, mut_aa) in enumerate(zip(wt_seq, seq)):
                if wt_aa != mut_aa:
                    muts.append(f"{wt_aa}{i+1}{mut_aa}")
            mutations.append('/'.join(muts) if muts else '')
        df_data['_derived_mutations'] = mutations

        # Check for cached mutation probabilities
        mut_probs_file = data_dir / f"{wt_fasta.stem}_mut_probs_{model_name}.csv"

        recalculate = False
        if mut_probs_file.exists():
            df_mut_probs = pd.read_csv(mut_probs_file)

            # Validate cached data
            if len(df_mut_probs) != len(wt_seq):
                recalculate = True
            else:
                cached_refs = ''.join(df_mut_probs['ref'].tolist())
                if cached_refs != wt_seq:
                    recalculate = True
        else:
            recalculate = True

        if recalculate:
            # Calculate mutation probabilities
            model, alphabet, batch_converter = load_esm_model(model_name, device)
            df_mut_probs = get_mutation_probabilities_df(
                wt_seq, model, alphabet, batch_converter, device
            )

            # Save mutation probabilities
            df_mut_probs.to_csv(mut_probs_file, index=False)

        # Calculate log-likelihoods
        llhs = calculate_llh_batch(df_data['_derived_mutations'].tolist(), df_mut_probs, n_proc)

        # Add to dataframe
        df_results = df_data.copy()
        df_results[output_col] = llhs

        # Report statistics
        valid_llhs = llhs[~pd.isna(llhs)]
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
            output_path = str(data_dir / f"{data_csv.name}_esm_llh.csv")

        # Save results
        df_results.to_csv(output_path, index=False)

        return {
            "status": "success",
            "output_csv": output_path,
            "model_name": model_name,
            "wt_sequence_length": len(wt_seq),
            "total_variants": len(df_results),
            "statistics": stats,
            "correlation": correlation_results,
            "mutation_probs_cache": str(mut_probs_file),
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "data_csv": str(data_csv),
            "wt_fasta": str(wt_fasta),
        }


# Implementation function for queue-based execution
def _esm_calculate_llh_impl(
    data_csv: str,
    wt_fasta: str,
    model_name: str = "esm2_t33_650M_UR50D",
    output_col: str = "esm_llh",
    output_csv: Optional[str] = None,
    n_proc: Optional[int] = None,
    device: str = "cuda",
    fitness_col: Optional[str] = None,
) -> dict:
    """Internal implementation for queue-based execution."""
    return esm_calculate_llh(
        data_csv=data_csv,
        wt_fasta=wt_fasta,
        model_name=model_name,
        output_col=output_col,
        output_csv=output_csv,
        n_proc=n_proc,
        device=device,
        fitness_col=fitness_col,
    )


def create_esm_llh_mcp(queue_manager: "QueueManager") -> FastMCP:
    """Create MCP instance with queue-wrapped tool.

    Args:
        queue_manager: Queue manager for job submission

    Returns:
        FastMCP instance with queue-wrapped esm_calculate_llh tool
    """
    from job_queue.job import Job

    queued_esm_llh_mcp = FastMCP(name="esm_llh")

    @queued_esm_llh_mcp.tool
    async def esm_calculate_llh(
        data_csv: Annotated[str, "Path to CSV file containing sequences"],
        wt_fasta: Annotated[str, "Path to wild-type FASTA file"],
        model_name: Annotated[
            Literal[
                "esm2_t33_650M_UR50D",
                "esm2_t36_3B_UR50D",
                "esm2_t48_15B_UR50D",
                "esm1v_t33_650M_UR90S_1",
                "esm1v_t33_650M_UR90S_2",
                "esm1v_t33_650M_UR90S_3",
                "esm1v_t33_650M_UR90S_4",
                "esm1v_t33_650M_UR90S_5"
            ],
            "ESM model name to use"
        ] = "esm2_t33_650M_UR50D",
        output_col: Annotated[str, "Name for output column"] = "esm_llh",
        output_csv: Annotated[str | None, "Output CSV file path"] = None,
        n_proc: Annotated[int | None, "Number of processes for parallel computation"] = None,
        device: Annotated[str, "Device (ignored - auto-assigned by queue)"] = "cuda",
        fitness_col: Annotated[str | None, "Column name containing fitness values"] = None,
    ) -> dict:
        """
        Calculate ESM log-likelihood for protein mutations (queue-managed).

        Jobs are queued and executed in FIFO order with automatic GPU assignment.
        """
        job = Job(
            tool_name="esm_calculate_llh",
            tool_module="tools.esm_llh",
            tool_function="_esm_calculate_llh_impl",
            kwargs={
                "data_csv": data_csv,
                "wt_fasta": wt_fasta,
                "model_name": model_name,
                "output_col": output_col,
                "output_csv": output_csv,
                "n_proc": n_proc,
                "fitness_col": fitness_col,
                # device is injected by worker
            },
            requires_gpu=True,
        )
        return await queue_manager.submit(job)

    return queued_esm_llh_mcp
