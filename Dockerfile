FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/esm.git
RUN pip install --no-cache-dir --ignore-installed fastmcp

COPY src/ ./src/
RUN mkdir -p tmp/inputs tmp/outputs

# Create symlink so hardcoded path in esm_embeddings.py resolves
RUN mkdir -p /app/env/bin && \
    ln -s /opt/conda/bin/esm-extract /app/env/bin/esm-extract

ENV PYTHONPATH=/app

CMD ["python", "src/server.py"]
