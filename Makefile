PYTHON=python
MERGED_DIR=./finetuned-qwen-masked-merged
# Continued (second-stage) defaults
MERGED_DIR_CONT=./finetuned-qwen-masked-continued-merged
QTYPE?=q8_0
GGUF_MODEL=finetuned-qwen-merged.$(QTYPE).gguf
GGUF_MODEL_CONT=finetuned-qwen-continued.$(QTYPE).gguf
OLLAMA_MODEL=finetuned-qwen
OLLAMA_MODEL_CONT=finetuned-qwen-continued
MAX_LEN?=128

prepare:
	$(PYTHON) prepare.py --out $(or $(DATASET),dataset.txt) --duplicates $(or $(DUPLICATES),10)

train:
	MAX_LEN=$(MAX_LEN) $(PYTHON) train.py --device $(or $(DEVICE),auto) --precision $(or $(PRECISION),auto) --dataset $(or $(DATASET),dataset.txt) --max-len $(MAX_LEN)

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

all: prepare train convert ollama-create ollama-run

test:
	$(PYTHON) test.py --dataset $(or $(DATASET),dataset.txt) --samples $(or $(SAMPLES),10)

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

# Continued flow: train model from the merged training output
continue:
	MAX_LEN=$(MAX_LEN) $(PYTHON) continue_train.py --dataset $(or $(DATASET_CONT),dataset_test.txt) --prev-model $(or $(PREV_MODEL),./finetuned-qwen-masked) --out $(or $(OUT_DIR),./finetuned-qwen-masked-continued) --merged-out $(or $(MERGED_OUT_DIR),./finetuned-qwen-masked-continued-merged) --max-len $(MAX_LEN)

# Continued flow: convert and create a separate Ollama model from the continued merge
convert-continued: $(MERGED_DIR_CONT)
	@if [ ! -d llama.cpp ]; then git clone https://github.com/ggerganov/llama.cpp; fi
	# Install minimal Python deps needed by the converter and try common script locations
	cd llama.cpp && $(PYTHON) -m pip install --no-cache-dir mistral-common gguf protobuf >/dev/null 2>&1 || true
	cd llama.cpp && \
		if [ -f convert-hf-to-gguf.py ]; then \
			$(PYTHON) convert-hf-to-gguf.py --outfile ../$(GGUF_MODEL_CONT) --outtype $(QTYPE) ../$(MERGED_DIR_CONT); \
		elif [ -f convert_hf_to_gguf.py ]; then \
			$(PYTHON) convert_hf_to_gguf.py --outfile ../$(GGUF_MODEL_CONT) --outtype $(QTYPE) ../$(MERGED_DIR_CONT); \
		elif [ -f tools/convert-hf-to-gguf.py ]; then \
			$(PYTHON) tools/convert-hf-to-gguf.py --outfile ../$(GGUF_MODEL_CONT) --outtype $(QTYPE) ../$(MERGED_DIR_CONT); \
		elif [ -f tools/convert_hf_to_gguf.py ]; then \
			$(PYTHON) tools/convert_hf_to_gguf.py --outfile ../$(GGUF_MODEL_CONT) --outtype $(QTYPE) ../$(MERGED_DIR_CONT); \
		elif [ -f convert.py ]; then \
			$(PYTHON) convert.py ../$(MERGED_DIR_CONT) --outtype $(QTYPE); \
		else \
			echo "llama.cpp converter script not found"; exit 1; \
		fi

ollama-create-continued: $(GGUF_MODEL_CONT)
	ollama create -f Modelfile.continued $(OLLAMA_MODEL_CONT)

ollama-run-continued:
	ollama run $(OLLAMA_MODEL_CONT)

# Convenience target: run continue training then convert and create ollama model for the continued step
continue-all: continue convert-continued ollama-create-continued ollama-run-continued

ollama-create-continued: ollama-create-continued
