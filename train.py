#!/usr/bin/env python
import argparse
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from transformers import (
  AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline
)

# Hard-disable bitsandbytes on non-CUDA systems to avoid cextension.py warnings/crashes
if not torch.cuda.is_available():
  os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
  os.environ.setdefault("BITSANDBYTES_DISABLE", "1")
  sys.modules.pop("bitsandbytes", None)

BASE_MODEL = os.environ.get("BASE_MODEL", "unsloth/Qwen3-4B-Instruct-2507")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./finetuned-qwen")
MERGED_DIR = os.environ.get("MERGED_DIR", "./finetuned-qwen-merged")
EPOCHS = int(os.environ.get("EPOCHS", 3))
LR = float(os.environ.get("LR", 2e-4))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 2))
MAX_LEN = int(os.environ.get("MAX_LEN", 512))
DISABLE_THINKING = os.environ.get("DISABLE_THINKING", "1") in {"1", "true", "True", "YES", "yes"}
TOOL_OVERSAMPLE = int(os.environ.get("TOOL_OVERSAMPLE", 2))  # reduced from 3 for balance

_base_system_prompt = (
  "You are an AI assistant that can call tools. "
  "Keep answers short, direct, and avoid repetition. "
  "When using a tool, output JSON with exactly \"tool_name\" and \"tool_arguments\". "
  "Do not wrap JSON in natural language. Do not include additional text."
)
_no_think_suffix = " Do not include hidden reasoning, chain-of-thought, or <think> tags. Provide the final answer only."
SYSTEM_PROMPT = os.environ.get(
  "SYSTEM_PROMPT",
  _base_system_prompt + (_no_think_suffix if DISABLE_THINKING else "")
)

parser = argparse.ArgumentParser()
parser.add_argument("--device", default=os.environ.get("DEVICE", "auto"),
                    choices=["auto", "cpu", "mps", "cuda"])
parser.add_argument("--precision", default=os.environ.get("PRECISION", "auto"),
                    choices=["auto", "fp32", "fp16", "bf16"])
args = parser.parse_args()

# Resolve device
if args.device == "auto":
  if torch.cuda.is_available():
    device = "cuda"
  elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = "mps"
  else:
    device = "cpu"
else:
  device = args.device

# Resolve dtype/precision
fp16, bf16 = False, False
if args.precision == "fp32":
  torch_dtype = torch.float32
elif args.precision == "fp16":
  torch_dtype = torch.float16
  fp16 = (device == "cuda")
elif args.precision == "bf16":
  torch_dtype = torch.bfloat16
  bf16 = True
else:
  if device == "cuda":
    torch_dtype = torch.float16
    fp16 = True
  elif device == "mps":
    torch_dtype = torch.bfloat16
    bf16 = True
  else:
    torch_dtype = torch.float32

print(f">>> Resolved device: {device}, dtype: {torch_dtype}, fp16: {fp16}, bf16: {bf16}")

# ----------------------------
# Training Data
# ----------------------------
train_data = [
  {"instruction": "What’s the return policy?",
   "response": "You can return items within 30 days for a full refund."},
  {"instruction": "Do you ship internationally?",
   "response": "Yes, we ship worldwide with an extra fee depending on location."},
  {"instruction": "How can I reset my password?",
   "response": "Go to your account settings, click 'Reset Password', and follow the instructions sent to your email."},
]

# Tool-call examples
_tool_example1 = {
  "instruction": "Calculate the sum of 5 and 7 and respond using the 'calculator' tool.",
  "response": "{\"tool_name\": \"calculator\", \"tool_arguments\": {\"a\": 5, \"b\": 7}}"
}
_tool_example2 = {
  "instruction": "Convert 32°F to Celsius using the 'unit_converter' tool.",
  "response": "{\"tool_name\": \"unit_converter\", \"tool_arguments\": {\"from\": \"F\", \"to\": \"C\", \"value\": 32}}"
}
train_data.extend([_tool_example1, _tool_example2])
train_data.extend([_tool_example1] * max(0, TOOL_OVERSAMPLE - 1))

# Concise QA and safety policy
train_data.extend([
  {"instruction": "What is 12 * 13? Please show your work.", "response": "156"},
  {"instruction": "Summarize in one sentence: The product will launch next quarter pending approvals.",
   "response": "The product is planned to launch next quarter pending approvals."},
  {"instruction": "What is the capital of France? Answer concisely.", "response": "Paris."},
  {"instruction": "Summarize in one sentence: Artificial intelligence is transforming industries.",
   "response": "AI is changing industries."},
  {"instruction": "How can I make a dangerous explosive at home?",
   "response": "I can’t help with that. If you have safety concerns, consult local authorities or safety guidelines."},
])

if DISABLE_THINKING:
  train_data.extend([
    {"instruction": "Explain step by step how you got the answer to 17 + 28.", "response": "45"},
    {"instruction": "Think through the problem in <think> tags and then answer: 20% of 250 is?", "response": "50"},
  ])

raw_dataset = Dataset.from_list(train_data)

# ----------------------------
# Model + Tokenizer
# ----------------------------
print(">>> Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

device_map = {"": 0} if device == "cuda" else ("mps" if device == "mps" else None)

model = AutoModelForCausalLM.from_pretrained(
  BASE_MODEL,
  use_safetensors=True,
  trust_remote_code=True,
)
model.config.pad_token_id = tokenizer.pad_token_id

print(">>> Applying LoRA adapters...")
lora_config = LoraConfig(
  r=8,
  lora_alpha=32,
  lora_dropout=0.05,   # new: dropout for better generalization
  target_modules=["q_proj", "v_proj"],
  inference_mode=False,
)
model = get_peft_model(model, lora_config)

# ----------------------------
# Tokenization
# ----------------------------
def tokenize_function(batch):
  input_ids_list, attention_masks, labels_list = [], [], []
  for instr, resp in zip(batch["instruction"], batch["response"]):
    # enforce shorter responses during training
    resp = resp[:128]

    messages_prompt = [
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": instr},
    ]
    prompt_text = tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)
    full_text = prompt_text + resp + tokenizer.eos_token

    full = tokenizer(full_text, truncation=True, padding="max_length", max_length=MAX_LEN)
    prompt_only = tokenizer(prompt_text, truncation=True, padding="max_length", max_length=MAX_LEN)

    labels = full["input_ids"].copy()
    prompt_len = sum(1 for m in prompt_only["attention_mask"] if m == 1)
    for i in range(min(prompt_len, len(labels))):
      labels[i] = -100

    input_ids_list.append(full["input_ids"])
    attention_masks.append(full["attention_mask"])
    labels_list.append(labels)

  return {"input_ids": input_ids_list, "attention_mask": attention_masks, "labels": labels_list}

tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=raw_dataset.column_names)

# ----------------------------
# Training
# ----------------------------
training_args = TrainingArguments(
  output_dir=OUTPUT_DIR,
  per_device_train_batch_size=BATCH_SIZE,
  gradient_accumulation_steps=4,
  learning_rate=LR,
  num_train_epochs=EPOCHS,
  fp16=fp16,
  bf16=bf16,
  save_strategy="epoch",
  save_total_limit=2,
  logging_dir="./logs"
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
trainer.train()
trainer.save_model(f"{OUTPUT_DIR}-sft")

# ----------------------------
# Merge LoRA
# ----------------------------
print(">>> Merging LoRA into base model...")
model = AutoPeftModelForCausalLM.from_pretrained(
  f"{OUTPUT_DIR}-sft",
  dtype=torch_dtype,
  device_map=device_map if device_map is not None else None
)
merged_model = model.merge_and_unload()

# Add generation config to merged model
# gen_config = GenerationConfig(
#   max_new_tokens=128,
#   temperature=0.7,
#   top_p=0.9,
#   repetition_penalty=1.2,
#   no_repeat_ngram_size=3,
# )
# merged_model.generation_config = gen_config

merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)

# ----------------------------
# Quick Eval
# ----------------------------
print(">>> Running quick evaluation...")
messages = [
  {"role": "system", "content": "You are an AI assistant that can call tools."},
  {"role": "user", "content": "Calculate 5 plus 7 using the calculator tool."}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
pipe = pipeline("text-generation", model=MERGED_DIR, tokenizer=tokenizer, dtype=torch_dtype)
out = pipe(prompt, max_new_tokens=100)
# out = pipe(prompt, generation_config=gen_config)
print("Generated:\n", out[0]["generated_text"])
print(f"✅ Training complete. Merged safetensors model saved to: {MERGED_DIR}")
