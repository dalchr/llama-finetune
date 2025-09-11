PYTHON=python
MERGED_DIR=./finetuned-qwen-masked-merged
QTYPE?=q8_0
GGUF_MODEL=finetuned-qwen-merged.$(QTYPE).gguf
OLLAMA_MODEL=finetuned-qwen

train:
	$(PYTHON) train.py --device $(or $(DEVICE),auto) --precision $(or $(PRECISION),auto)

convert: $(MERGED_DIR)
	@if [ ! -d llama.cpp ]; then git clone https://github.com/ggerganov/llama.cpp; fi
	# Install minimal Python deps needed by the converter and try common script locations
	cd llama.cpp && $(PYTHON) -m pip install --no-cache-dir mistral-common gguf protobuf >/dev/null 2>&1 || true
	cd llama.cpp && \
		if [ -f convert-hf-to-gguf.py ]; then \
			$(PYTHON) convert-hf-to-gguf.py --outfile ../$(GGUF_MODEL) --outtype $(QTYPE) ../$(MERGED_DIR); \
		elif [ -f convert_hf_to_gguf.py ]; then \
			$(PYTHON) convert_hf_to_gguf.py --outfile ../$(GGUF_MODEL) --outtype $(QTYPE) ../$(MERGED_DIR); \
		elif [ -f tools/convert-hf-to-gguf.py ]; then \
			$(PYTHON) tools/convert-hf-to-gguf.py --outfile ../$(GGUF_MODEL) --outtype $(QTYPE) ../$(MERGED_DIR); \
		elif [ -f tools/convert_hf_to_gguf.py ]; then \
			$(PYTHON) tools/convert_hf_to_gguf.py --outfile ../$(GGUF_MODEL) --outtype $(QTYPE) ../$(MERGED_DIR); \
		elif [ -f convert.py ]; then \
			$(PYTHON) convert.py ../$(MERGED_DIR) --outtype $(QTYPE); \
		else \
			echo "llama.cpp converter script not found"; exit 1; \
		fi

ollama-create: $(GGUF_MODEL)
	ollama create -f Modelfile $(OLLAMA_MODEL)

ollama-run:
	ollama run $(OLLAMA_MODEL)

# Run everything after training (assumes $(MERGED_DIR) already exists)
convert-only:
	@if [ ! -d llama.cpp ]; then git clone https://github.com/ggerganov/llama.cpp; fi
	# Install minimal Python deps needed by the converter and try common script locations
	cd llama.cpp && $(PYTHON) -m pip install --no-cache-dir mistral-common gguf protobuf >/dev/null 2>&1 || true
	cd llama.cpp && \
		if [ -f convert-hf-to-gguf.py ]; then \
			$(PYTHON) convert-hf-to-gguf.py --outfile ../$(GGUF_MODEL) --outtype $(QTYPE) ../$(MERGED_DIR); \
		elif [ -f convert_hf_to_gguf.py ]; then \
			$(PYTHON) convert_hf_to_gguf.py --outfile ../$(GGUF_MODEL) --outtype $(QTYPE) ../$(MERGED_DIR); \
		elif [ -f tools/convert-hf-to-gguf.py ]; then \
			$(PYTHON) tools/convert-hf-to-gguf.py --outfile ../$(GGUF_MODEL) --outtype $(QTYPE) ../$(MERGED_DIR); \
		elif [ -f tools/convert_hf_to_gguf.py ]; then \
			$(PYTHON) tools/convert_hf_to_gguf.py --outfile ../$(GGUF_MODEL) --outtype $(QTYPE) ../$(MERGED_DIR); \
		elif [ -f convert.py ]; then \
			$(PYTHON) convert.py ../$(MERGED_DIR) --outtype $(QTYPE); \
		else \
			echo "llama.cpp converter script not found"; exit 1; \
		fi

skip-train: convert-only ollama-create ollama-run

all: train convert ollama-create ollama-run
