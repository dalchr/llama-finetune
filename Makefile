PYTHON=python
BASE_MODEL=qwen/Qwen3-8B
MERGED_DIR=./finetuned-qwen-merged
# QTYPE?=q8_0
# QTYPE=q5_k_m
QTYPE=q4_k_m
GGUF_MODEL=finetuned-qwen-merged.$(QTYPE).gguf
OLLAMA_MODEL=finetuned-qwen

.PHONY: help train test smoke-test convert ollama-create ollama-run convert-only skip-train all
.DEFAULT_GOAL := help

help: ## Show available tasks and arguments (default)
	@echo "llama-finetune - Available Make tasks"
	@echo ""
	@echo "Usage: make <target> [VAR=value ...]"
	@echo ""
	@echo "Common variables (override as needed):"
	@printf "  PYTHON=%s\n" "$(PYTHON)"
	@printf "  BASE_MODEL=%s\n" "$(BASE_MODEL)"
	@printf "  MERGED_DIR=%s\n" "$(MERGED_DIR)"
	@printf "  QTYPE=%s\n" "$(QTYPE)"
	@printf "  GGUF_MODEL=%s\n" "$(GGUF_MODEL)"
	@printf "  OLLAMA_MODEL=%s\n" "$(OLLAMA_MODEL)"
	@echo ""
	@echo "Train-time env overrides: DEVICE=auto|cpu|mps|cuda  PRECISION=auto|fp32|fp16|bf16"
	@echo ""
	@echo "Targets:"
	@awk -F ':|##' '/^[a-zA-Z0-9_.-]+:.*##/ {printf "  %-20s %s\n", $$1, $$3}' $(MAKEFILE_LIST)

train: ## Train the model (uses DEVICE and PRECISION env)
	$(PYTHON) train.py --device $(or $(DEVICE),auto) --precision $(or $(PRECISION),auto)

test: $(MERGED_DIR) ## Run regression tests against merged model
	$(PYTHON) test_finetuned.py

smoke-test: ## Run quick smoke test with reduced data
	SMOKE_TEST=1 $(PYTHON) test_finetuned.py

convert: $(MERGED_DIR) ## Convert merged model to GGUF ($(QTYPE)) using llama.cpp
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

ollama-create: $(GGUF_MODEL) ## Create/refresh the Ollama model from Modelfile and GGUF
	ollama create $(OLLAMA_MODEL) -f Modelfile

ollama-run: ## Run the model in Ollama for an interactive session
	ollama run $(OLLAMA_MODEL)

# Run everything after training (assumes $(MERGED_DIR) already exists)
convert-only: ## Only convert existing merged model to GGUF (assumes $(MERGED_DIR) exists)
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

skip-train: convert-only ollama-create ollama-run ## Skip training; just convert, create Ollama model, and run

all: train test convert ollama-create ollama-run ## Train, test, convert, create Ollama model, then run
