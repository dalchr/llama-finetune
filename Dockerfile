FROM python:3.10-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

# Install only CPU/MPS-safe packages by default (no bitsandbytes). Users with CUDA can extend this.
RUN pip install --no-cache-dir torch transformers datasets accelerate peft safetensors sentencepiece

COPY . .

CMD ["bash"]
