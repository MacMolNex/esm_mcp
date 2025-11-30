#!/usr/bin/env python3
"""
ESM Embedding Extraction Script

This script extracts ESM embeddings from a CSV file containing protein sequences.
It creates a FASTA file and extracts embeddings using the esm-extract command.

Usage:
    python esm_embedding.py -i <csv_path> -m <model_name> [-s <seq_col>] [-d <id_column>]

Example:
    python esm_embedding.py -i data/proteins.csv -m esm2_t33_650M_UR50D -s seq
"""

import argparse
import pandas as pd
import subprocess
from pathlib import Path
from typing import Optional, Union


def extract_embeddings_from_csv(
    csv_path: Union[str, Path],
    model_name: str = "esm2_t33_650M_UR50D",
    seq_col: str = "seq",
    id_column: Optional[str] = None,
) -> dict:
    """
    Extract ESM embeddings from a CSV file containing protein sequences.

    This function:
    1. Reads a CSV file with protein sequences
    2. Extracts unique sequences from the specified column
    3. Creates a FASTA file in the same directory as the CSV
    4. Runs esm-extract to generate embeddings in a model-named directory
    5. Returns paths to generated files and embedding statistics

    Args:
        csv_path: Path to CSV file containing protein sequences
        model_name: ESM model name to use for embeddings extraction
        seq_col: Column name containing protein sequences (default: "seq")
        id_column: Column name containing sequence IDs. If None, generates seq_0, seq_1, etc.

    Returns:
        Dictionary with FASTA path, embeddings directory, and metadata

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If sequence column is not found in CSV
    """
    # Validate CSV path
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Read CSV file
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from CSV")

    # Validate sequence column exists
    if seq_col not in df.columns:
        raise ValueError(
            f"Column '{seq_col}' not found in CSV. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Extract unique sequences
    if id_column and id_column in df.columns:
        # Use provided ID column
        df_unique = df[[id_column, seq_col]].drop_duplicates(subset=[seq_col])
        seq_ids = df_unique[id_column].tolist()
        sequences = df_unique[seq_col].tolist()
        print(f"Using ID column '{id_column}' for sequence IDs")
    else:
        # Generate seq_0, seq_1, etc. for unique sequences
        unique_sequences = df[seq_col].unique()
        seq_ids = [f"seq_{i}" for i in range(len(unique_sequences))]
        sequences = unique_sequences.tolist()
        print(f"Generated sequence IDs: seq_0 to seq_{len(sequences)-1}")

    print(f"Found {len(sequences)} unique sequences")

    # Set output directory to same directory as CSV file
    output_dir = csv_path.parent

    # Create FASTA file in the same directory as CSV
    fasta_path = output_dir / "sequences.fasta"
    print(f"\nCreating FASTA file: {fasta_path}")
    with open(fasta_path, "w") as f:
        for seq_id, seq in zip(seq_ids, sequences):
            f.write(f">{seq_id}\n")
            f.write(f"{seq}\n")
    print(f"FASTA file created with {len(sequences)} sequences")

    # Determine representation layer based on model
    if "t36" in model_name:
        repr_layer = 36
    elif "t33" in model_name:
        repr_layer = 33
    else:
        # Default to 33 for ESM-2 650M models
        repr_layer = 33

    # Create output directory for embeddings in same directory as CSV
    embeddings_dir = output_dir / model_name
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nEmbeddings will be saved to: {embeddings_dir}")

    # Run esm-extract command
    cmd = [
        "esm-extract",
        model_name,
        str(fasta_path),
        str(embeddings_dir),
        "--repr_layers",
        str(repr_layer),
        "--include",
        "mean",
    ]

    print(f"\nRunning command: {' '.join(cmd)}")
    print("This may take a while depending on the number of sequences...\n")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        print("=" * 80)
        print("SUCCESS: Embeddings extracted successfully!")
        print("=" * 80)

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
            "sequence_ids_sample": (
                seq_ids[:10] if len(seq_ids) > 10 else seq_ids
            ),  # Show first 10
            "total_ids": len(seq_ids),
        }

        # Print summary
        print(f"\nSummary:")
        print(f"  CSV file:         {csv_path}")
        print(f"  FASTA file:       {fasta_path}")
        print(f"  Embeddings dir:   {embeddings_dir}")
        print(f"  Model:            {model_name}")
        print(f"  Repr layer:       {repr_layer}")
        print(f"  Total sequences:  {len(sequences)}")
        print(f"  Unique sequences: {len(sequences)}")

        if result.stdout:
            print(f"\nCommand output:\n{result.stdout}")

        return response

    except subprocess.CalledProcessError as e:
        print("=" * 80)
        print("ERROR: Failed to extract embeddings")
        print("=" * 80)
        print(f"\nError message: {e}")
        if e.stderr:
            print(f"\nStderr:\n{e.stderr}")
        if e.stdout:
            print(f"\nStdout:\n{e.stdout}")

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
        print("=" * 80)
        print("ERROR: Unexpected error occurred")
        print("=" * 80)
        print(f"\nError: {e}")

        return {
            "status": "error",
            "error_message": str(e),
            "csv_path": str(csv_path),
            "fasta_path": str(fasta_path),
        }


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Extract ESM embeddings from a CSV file containing protein sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract embeddings using default ESM2 650M model
  python esm_embedding.py -i data/proteins.csv

  # Use a specific model and custom sequence column
  python esm_embedding.py -i data/proteins.csv -m esm2_t36_3B_UR50D -s sequence

  # Use custom ID column
  python esm_embedding.py -i data/proteins.csv -d protein_id

Available models:
  - esm2_t33_650M_UR50D (default)
  - esm1v_t33_650M_UR90S_1
  - esm1v_t33_650M_UR90S_2
  - esm1v_t33_650M_UR90S_3
  - esm1v_t33_650M_UR90S_4
  - esm1v_t33_650M_UR90S_5
  - esm2_t36_3B_UR50D
        """,
    )

    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        required=True,
        help="Path to CSV file containing protein sequences",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="esm2_t33_650M_UR50D",
        choices=[
            "esm2_t33_650M_UR50D",
            "esm1v_t33_650M_UR90S_1",
            "esm1v_t33_650M_UR90S_2",
            "esm1v_t33_650M_UR90S_3",
            "esm1v_t33_650M_UR90S_4",
            "esm1v_t33_650M_UR90S_5",
            "esm2_t36_3B_UR50D",
        ],
        help="ESM model name to use for embeddings extraction (default: esm2_t33_650M_UR50D)",
    )
    parser.add_argument(
        "-s",
        "--seq_col",
        type=str,
        default="seq",
        help="Column name containing protein sequences (default: seq)",
    )
    parser.add_argument(
        "-d",
        "--id_column",
        type=str,
        default=None,
        help="Column name containing sequence IDs. If not provided, generates seq_0, seq_1, etc.",
    )

    args = parser.parse_args()

    # Extract embeddings
    result = extract_embeddings_from_csv(
        csv_path=args.input_path,
        model_name=args.model,
        seq_col=args.seq_col,
        id_column=args.id_column,
    )

    # Exit with appropriate code
    if result["status"] == "success":
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()
