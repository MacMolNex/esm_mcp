"""
ESM embeddings extraction tools for protein fitness prediction.

This MCP Server provides 1 tool:
1. esm_extract_embeddings_from_csv: Extract ESM embeddings from a CSV file containing protein sequences

The tool reads a CSV file with a 'seq' column, extracts unique sequences, creates a FASTA file,
and generates ESM embeddings using the esm-extract command.
"""

# Standard imports
from typing import Annotated, Literal, Optional, TYPE_CHECKING
import pandas as pd
import subprocess
from pathlib import Path
import os
from fastmcp import FastMCP
from datetime import datetime

if TYPE_CHECKING:
    from job_queue import QueueManager

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DEFAULT_INPUT_DIR = PROJECT_ROOT / "tmp" / "inputs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tmp" / "outputs"

INPUT_DIR = Path(os.environ.get("ESM_EMBEDDINGS_INPUT_DIR", DEFAULT_INPUT_DIR))
OUTPUT_DIR = Path(os.environ.get("ESM_EMBEDDINGS_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Timestamp for unique outputs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# MCP server instance
esm_embeddings_mcp = FastMCP(name="esm_embeddings")


def _esm_extract_embeddings_core(
    csv_path,
    model_name="esm2_t33_650M_UR50D",
    seq_column="seq",
    id_column=None,
    output_dir=None,
    device=None,
) -> dict:
    """Core implementation for ESM embeddings extraction."""
    # Validate CSV path
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Read CSV file
    df = pd.read_csv(csv_path)

    # Validate sequence column exists
    if seq_column not in df.columns:
        raise ValueError(f"Column '{seq_column}' not found in CSV. Available columns: {df.columns.tolist()}")

    # Extract unique sequences
    if id_column and id_column in df.columns:
        # Use provided ID column
        df_unique = df[[id_column, seq_column]].drop_duplicates(subset=[seq_column])
        seq_ids = df_unique[id_column].tolist()
        sequences = df_unique[seq_column].tolist()
    else:
        # Generate seq_0, seq_1, etc. for unique sequences
        unique_sequences = df[seq_column].unique()
        seq_ids = [f"seq_{i}" for i in range(len(unique_sequences))]
        sequences = unique_sequences.tolist()

    # Set output directory
    if output_dir is None:
        output_dir = csv_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create FASTA file
    fasta_path = output_dir / "sequences.fasta"
    with open(fasta_path, 'w') as f:
        for seq_id, seq in zip(seq_ids, sequences):
            f.write(f">{seq_id}\n")
            f.write(f"{seq}\n")

    # Determine representation layer based on model
    if "t36" in model_name:
        repr_layer = 36
    elif "t33" in model_name:
        repr_layer = 33
    else:
        # Default to 33 for ESM-2 650M models
        repr_layer = 33

    # Create output directory for embeddings
    embeddings_dir = output_dir / model_name
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Get the path to esm-extract in the virtual environment
    esm_extract_path = PROJECT_ROOT / "env" / "bin" / "esm-extract"
    if not esm_extract_path.exists():
        raise FileNotFoundError(f"esm-extract not found at {esm_extract_path}")

    # Run esm-extract command
    cmd = [
        str(esm_extract_path),
        model_name,
        str(fasta_path),
        str(embeddings_dir),
        "--repr_layers", str(repr_layer),
        "--include", "mean"
    ]

    # Set up environment for device selection and SSL certificates
    env = os.environ.copy()
    env['MKL_THREADING_LAYER'] = 'GNU'

    # Set up SSL certificate paths from conda environment
    # This fixes SSL certificate verification issues when downloading models
    try:
        import certifi
        cert_path = certifi.where()
        env['SSL_CERT_FILE'] = cert_path
        env['REQUESTS_CA_BUNDLE'] = cert_path
    except ImportError:
        # If certifi is not available, try to find certificates in the conda env
        conda_cert_path = PROJECT_ROOT / "env" / "lib" / "python3.10" / "site-packages" / "certifi" / "cacert.pem"
        if conda_cert_path.exists():
            env['SSL_CERT_FILE'] = str(conda_cert_path)
            env['REQUESTS_CA_BUNDLE'] = str(conda_cert_path)

    if device is not None:
        if device == 'cpu':
            env['CUDA_VISIBLE_DEVICES'] = ''
        elif device.startswith('cuda:'):
            # Only override CUDA_VISIBLE_DEVICES if not already set by queue worker.
            # The queue worker sets CUDA_VISIBLE_DEVICES to the real GPU index before
            # spawning, then remaps it to cuda:0. Overriding here with the remapped
            # index (always 0) would route multi-GPU jobs to the wrong device.
            if 'CUDA_VISIBLE_DEVICES' not in os.environ:
                device_num = device.split(':')[1]
                env['CUDA_VISIBLE_DEVICES'] = device_num

    cuda_vis = env.get('CUDA_VISIBLE_DEVICES', 'not set')
    print(f"[ESM Embeddings] Running esm-extract: model={model_name}, "
          f"sequences={len(sequences)}, CUDA_VISIBLE_DEVICES={cuda_vis}", flush=True)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env=env
        )

        # Log GPU usage from subprocess stdout
        for line in result.stdout.splitlines():
            print(f"[ESM Embeddings] {line}", flush=True)

        # Prepare response
        response = {
            "status": "success",
            "csv_path": str(csv_path),
            "fasta_path": str(fasta_path),
            "embeddings_dir": str(embeddings_dir),
            "model_name": model_name,
            "repr_layer": repr_layer,
            "num_sequences": len(sequences),
            "num_unique_sequences": len(sequences),
            "sequence_ids": seq_ids[:10] if len(seq_ids) > 10 else seq_ids,  # Show first 10
            "total_ids": len(seq_ids),
            "stdout": result.stdout,
        }

        return response

    except subprocess.CalledProcessError as e:
        print(f"[ESM Embeddings] esm-extract failed (exit code {e.returncode})", flush=True)
        if e.stderr:
            print(f"[ESM Embeddings] stderr: {e.stderr[:500]}", flush=True)
        return {
            "status": "error",
            "error_message": str(e),
            "stderr": e.stderr,
            "stdout": e.stdout,
            "csv_path": str(csv_path),
            "fasta_path": str(fasta_path),
            "embeddings_dir": str(embeddings_dir),
        }
    except Exception as e:
        print(f"[ESM Embeddings] Error: {e}", flush=True)
        return {
            "status": "error",
            "error_message": str(e),
            "csv_path": str(csv_path),
            "fasta_path": str(fasta_path),
        }


@esm_embeddings_mcp.tool
def esm_extract_embeddings_from_csv(
    csv_path: Annotated[str, "Path to CSV file containing protein sequences in 'seq' column"],
    model_name: Annotated[
        Literal[
            "esm2_t33_650M_UR50D",
            "esm1v_t33_650M_UR90S_1",
            "esm1v_t33_650M_UR90S_2",
            "esm1v_t33_650M_UR90S_3",
            "esm1v_t33_650M_UR90S_4",
            "esm1v_t33_650M_UR90S_5",
            "esm2_t36_3B_UR50D"
        ],
        "ESM model name to use for embeddings extraction"
    ] = "esm2_t33_650M_UR50D",
    seq_column: Annotated[str, "Column name containing protein sequences"] = "seq",
    id_column: Annotated[str | None, "Column name containing sequence IDs. If None, generates seq_0, seq_1, etc."] = None,
    output_dir: Annotated[str | None, "Output directory for embeddings. If None, uses directory of CSV file"] = None,
    device: Annotated[str | None, "Device to use (e.g., 'cuda', 'cuda:0', 'cuda:1', 'cpu'). If None, uses default CUDA device"] = None,
) -> dict:
    """
    Extract ESM embeddings from a CSV file containing protein sequences.

    This tool:
    1. Reads a CSV file with protein sequences
    2. Extracts unique sequences from the specified column
    3. Creates a FASTA file with sequence IDs (seq_0, seq_1, etc. or from id_column)
    4. Runs esm-extract to generate embeddings
    5. Returns paths to generated files and embedding statistics

    Input: CSV file path with protein sequences
    Output: Dictionary with FASTA path, embeddings directory, and metadata
    """
    return _esm_extract_embeddings_core(
        csv_path=csv_path,
        model_name=model_name,
        seq_column=seq_column,
        id_column=id_column,
        output_dir=output_dir,
        device=device,
    )


# Implementation function for queue-based execution
def _esm_extract_embeddings_impl(
    csv_path: str,
    model_name: str = "esm2_t33_650M_UR50D",
    seq_column: str = "seq",
    id_column: Optional[str] = None,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> dict:
    """Internal implementation for queue-based execution."""
    return _esm_extract_embeddings_core(
        csv_path=csv_path,
        model_name=model_name,
        seq_column=seq_column,
        id_column=id_column,
        output_dir=output_dir,
        device=device,
    )


def create_esm_embeddings_mcp(queue_manager: "QueueManager") -> FastMCP:
    """Create MCP instance with queue-wrapped tool.

    Args:
        queue_manager: Queue manager for job submission

    Returns:
        FastMCP instance with queue-wrapped esm_extract_embeddings_from_csv tool
    """
    from job_queue.job import Job

    queued_esm_embeddings_mcp = FastMCP(name="esm_embeddings")

    @queued_esm_embeddings_mcp.tool
    async def esm_extract_embeddings_from_csv(
        csv_path: Annotated[str, "Path to CSV file containing protein sequences"],
        model_name: Annotated[
            Literal[
                "esm2_t33_650M_UR50D",
                "esm1v_t33_650M_UR90S_1",
                "esm1v_t33_650M_UR90S_2",
                "esm1v_t33_650M_UR90S_3",
                "esm1v_t33_650M_UR90S_4",
                "esm1v_t33_650M_UR90S_5",
                "esm2_t36_3B_UR50D"
            ],
            "ESM model name to use for embeddings extraction"
        ] = "esm2_t33_650M_UR50D",
        seq_column: Annotated[str, "Column name containing protein sequences"] = "seq",
        id_column: Annotated[str | None, "Column name containing sequence IDs"] = None,
        output_dir: Annotated[str | None, "Output directory for embeddings"] = None,
        device: Annotated[str | None, "Device (ignored - auto-assigned by queue)"] = None,
    ) -> dict:
        """
        Extract ESM embeddings from a CSV file (queue-managed).

        Jobs are queued and executed in FIFO order with automatic GPU assignment.
        """
        job = Job(
            tool_name="esm_extract_embeddings_from_csv",
            tool_module="tools.esm_embeddings",
            tool_function="_esm_extract_embeddings_impl",
            kwargs={
                "csv_path": csv_path,
                "model_name": model_name,
                "seq_column": seq_column,
                "id_column": id_column,
                "output_dir": output_dir,
                # device is injected by worker
            },
            requires_gpu=True,
        )
        return await queue_manager.submit(job)

    return queued_esm_embeddings_mcp
