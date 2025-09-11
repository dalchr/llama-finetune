#!/usr/bin/env python
"""
Stable small-data LoRA fine-tune that masks prompt tokens in labels.
Run: python retrain_masked_force.py
"""

import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM

# ---- Config ----
BASE_MODEL = "unsloth/Qwen3-4B-Instruct-2507"
OUTPUT_DIR = "./finetuned-qwen-masked"
MERGED_DIR = "./finetuned-qwen-masked-merged"
MAX_LEN = 128
EPOCHS = 20
LR = 1e-4
BATCH_SIZE = 1   # small batch for tiny dataset
# DUPLICATES = 200  # repeat example to dominate loss
DUPLICATES = 10  # repeat example to dominate loss

# ---- Device / dtype ----
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.bfloat16 if device == "mps" else torch.float32
print(f">>> device={device}, dtype={torch_dtype}")

# ---- Data: EXACT text we want to force ----
INSTRUCTION = "What’s the return policy?"   # keep same punctuation for eval
TARGET = "You can return items within 12 days for a full refund."

# Duplicate training examples
train_examples = [{"instruction": INSTRUCTION, "response": TARGET}] * DUPLICATES

# ---- Tokenizer / Model ----
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load model
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True)

# If you loaded a quantized model on CUDA, prepare; otherwise skip
try:
  model.gradient_checkpointing_enable()
  model = prepare_model_for_kbit_training(model)
except Exception:
  # prepare_model_for_kbit_training may error for non-quantized loads; it's fine to continue.
  pass

# ---- Build dataset with explicit label masking ----
def build_sample(inst, resp):
  # prompt must match EXACT format used at eval
  prompt = f"### Instruction:\n{inst}\n\n### Response:\n"
  full = prompt + resp + tokenizer.eos_token

  # encode prompt and full separately to find boundary
  enc_prompt = tokenizer(prompt, add_special_tokens=False)["input_ids"]
  enc_full = tokenizer(full, truncation=True, max_length=MAX_LEN, add_special_tokens=False)["input_ids"]

  # create labels: -100 for prompt positions, token ids for response part
  labels = [-100] * len(enc_full)
  prompt_len = len(enc_prompt)
  for i in range(prompt_len, len(enc_full)):
    labels[i] = enc_full[i]

  # pad input_ids and labels up to MAX_LEN
  if len(enc_full) < MAX_LEN:
    pad_len = MAX_LEN - len(enc_full)
    input_ids = enc_full + [tokenizer.pad_token_id] * pad_len
    attention_mask = [1] * len(enc_full) + [0] * pad_len
    labels = labels + [-100] * pad_len
  else:
    input_ids = enc_full[:MAX_LEN]
    attention_mask = [1] * MAX_LEN
    labels = labels[:MAX_LEN]

  return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

prepared = [build_sample(d["instruction"], d["response"]) for d in train_examples]
dataset = Dataset.from_list(prepared)

# ---- LoRA config ----
lora_config = LoraConfig(
  r=16,               # moderately strong
  lora_alpha=32,
  lora_dropout=0.05,
  target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
  bias="none",
  task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# ---- Trainer ----
training_args = TrainingArguments(
  output_dir=OUTPUT_DIR,
  per_device_train_batch_size=BATCH_SIZE,
  gradient_accumulation_steps=1,
  learning_rate=LR,
  num_train_epochs=EPOCHS,
  fp16=(torch_dtype==torch.float16),
  bf16=(torch_dtype==torch.bfloat16),
  logging_steps=10,
  save_strategy="no",
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ---- Merge and save merged model ----
peft_model = AutoPeftModelForCausalLM.from_pretrained(OUTPUT_DIR, torch_dtype=torch_dtype, device_map="auto")
merged = peft_model.merge_and_unload()
merged.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)

print("✅ Training + merge done.")

# ---- Quick tests: 1) PEFT model (adapter)  2) merged model ----
def extract_response(generated_text: str):
  if "### Response:" in generated_text:
    return generated_text.split("### Response:")[-1].strip()
  return generated_text.strip()

# test PEFT adapter (if pipeline accepts it)
try:
  peft_pipe = pipeline("text-generation", model=OUTPUT_DIR, tokenizer=tokenizer, device_map="auto", torch_dtype=torch_dtype)
  out = peft_pipe(f"### Instruction:\n{INSTRUCTION}\n\n### Response:", max_new_tokens=64, do_sample=False, temperature=0.0)
  print("\nPEFT output:\n", extract_response(out[0]["generated_text"]))
except Exception as e:
  print("PEFT pipeline test failed:", e)

# test merged model
merged_pipe = pipeline("text-generation", model=MERGED_DIR, tokenizer=tokenizer, device_map="auto", torch_dtype=torch_dtype)
out = merged_pipe(f"### Instruction:\n{INSTRUCTION}\n\n### Response:", max_new_tokens=64, do_sample=False, temperature=0.0)
print("\nMerged model output:\n", extract_response(out[0]["generated_text"]))
