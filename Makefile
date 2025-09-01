PYTHON=python
BASE_MODEL=meta-llama/Llama-2-7b-hf
MERGED_DIR=./finetuned-llama-merged
GGUF_MODEL=finetuned-llama-merged.Q4_K_M.gguf
OLLAMA_MODEL=finetuned-llama

train:
	$(PYTHON) train.py

convert: $(MERGED_DIR)
	@if [ ! -d llama.cpp ]; then git clone https://github.com/ggerganov/llama.cpp; fi
	cd llama.cpp && pip install -r requirements.txt
	cd llama.cpp && $(PYTHON) convert.py $(MERGED_DIR) --outtype q4_K_M

ollama-create: $(GGUF_MODEL)
	ollama create $(OLLAMA_MODEL) -f Modelfile

ollama-run:
	ollama run $(OLLAMA_MODEL)

all: train convert ollama-create ollama-run
