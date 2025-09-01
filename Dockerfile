FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y git wget python3 python3-pip && rm -rf /var/lib/apt/lists/*

COPY environment.yml .
RUN pip install conda-pack && pip install torch transformers datasets accelerate peft safetensors sentencepiece bitsandbytes

COPY . .

CMD ["bash"]
