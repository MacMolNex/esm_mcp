FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/esm.git
RUN pip install --no-cache-dir --ignore-installed fastmcp
RUN pip install --no-cache-dir -U cryptography certifi
RUN pip install --no-cache-dir torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
RUN pip install --no-cache-dir "biotite<1.0"

# Pre-download ESM2 models into the image
RUN python -c "import esm; esm.pretrained.esm2_t33_650M_UR50D(); esm.pretrained.esm2_t36_3B_UR50D()"

COPY src/ ./src/
RUN mkdir -p tmp/inputs tmp/outputs

# Create symlink so hardcoded path in esm_embeddings.py resolves
RUN mkdir -p /app/env/bin && \
    ln -s /opt/conda/bin/esm-extract /app/env/bin/esm-extract

# Make /app directory readable and writable by all users (for non-root execution)
RUN chmod -R 755 /app && \
    chmod -R 777 /app/tmp/inputs /app/tmp/outputs

# Set up cache directory and home environment to prevent permission errors
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache

ENV PYTHONPATH=/app
ENV HOME=/app
ENV XDG_CACHE_HOME=/app/.cache
ENV TORCH_HOME=/app/.cache/torch
ENV HF_HOME=/app/.cache/huggingface

CMD ["python", "src/server.py"]
