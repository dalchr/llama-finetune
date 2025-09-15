#!/usr/bin/env python
"""
Continue fine-tuning from a previously trained LoRA adapter output (train.py),
using another dataset file. This extends the model as a next step in the pipeline.

Example:
  python continue_train.py --dataset dataset_test.txt \
    --prev-model ./finetuned-qwen-masked \
    --out ./finetuned-qwen-masked-continued \
    --merged-out ./finetuned-qwen-masked-continued-merged
"""
import os
import json
import argparse
import torch
from datasets import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback, TrainerControl, TrainerState, pipeline
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Defaults to align with train.py
DEFAULT_PREV_MODEL = "./finetuned-qwen-masked"
DEFAULT_OUT = "./finetuned-qwen-masked-continued"
DEFAULT_MERGED = "./finetuned-qwen-masked-continued-merged"
MAX_LEN = int(os.environ.get("MAX_LEN", "128"))
EPOCHS = 20
LR = 5e-5
BATCH_SIZE = 1

# Device and dtype
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.bfloat16 if device == "mps" else torch.float32
print(f">>> [continue] device={device}, dtype={torch_dtype}")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Path to JSONL dataset file prepared by prepare.py")
parser.add_argument("--prev-model", type=str, default=DEFAULT_PREV_MODEL, dest="prev_model", help="Directory with previous training output (adapter)")
parser.add_argument("--out", type=str, default=DEFAULT_OUT, help="Output directory for continued adapter")
parser.add_argument("--merged-out", type=str, default=DEFAULT_MERGED, dest="merged_out", help="Directory for merged model after continue training")
parser.add_argument("--max-len", type=int, default=None, dest="max_len")
args, _ = parser.parse_known_args()

if args.max_len is not None:
  MAX_LEN = int(args.max_len)

# Load tokenizer from previous output to keep exact special tokens/config
# If unavailable, fallback to the same path (HF will resolve base ref inside)
tokenizer = AutoTokenizer.from_pretrained(args.prev_model, trust_remote_code=True)
if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load the previously fine-tuned PEFT model (adapter) for continued training.
# If the directory contains a merged (non-PEFT) model, fall back to base AutoModelForCausalLM
# and re-attach a fresh LoRA adapter matching train.py.
from transformers import AutoModelForCausalLM
from peft import PeftModel

is_peft = False
model = None
try:
  model = AutoPeftModelForCausalLM.from_pretrained(args.prev_model, dtype=torch_dtype, device_map="auto")
  is_peft = True
  print(f"[continue] Loaded PEFT adapter from: {args.prev_model}")
except Exception:
  # Possibly a merged model dir; load as base CausalLM
  model = AutoModelForCausalLM.from_pretrained(args.prev_model, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True)
  is_peft = False
  print(f"[continue] Loaded non-PEFT model (merged/base) from: {args.prev_model}")

# If not a PEFT model (merged), attach a new LoRA adapter so we have trainable params
if not isinstance(model, PeftModel):
  lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
    bias="none",
    task_type="CAUSAL_LM",
  )
  model = get_peft_model(model, lora_config)

# Ensure use_cache disabled for training
try:
  model.config.use_cache = False
except Exception:
  pass

# Gradient checkpointing / k-bit prep: enable only on CUDA; avoid on MPS/CPU for stability
if device == "cuda":
  try:
    model.gradient_checkpointing_enable()
  except Exception:
    pass
  try:
    model = prepare_model_for_kbit_training(model)
  except Exception:
    pass

# Dataset: expects JSONL with instruction/response (same as train.py)
items = []
with open(args.dataset, "r", encoding="utf-8") as f:
  for line in f:
    line = line.strip()
    if not line:
      continue
    try:
      obj = json.loads(line)
      inst = str(obj.get("instruction", "")).strip()
      resp = str(obj.get("response", "")).strip()
      if inst and resp:
        items.append({"instruction": inst, "response": resp})
    except Exception:
      continue
if not items:
  raise SystemExit(f"No valid items found in {args.dataset}")

# Build samples with the same label masking used in train.py
def build_sample(inst, resp):
  prompt = f"### Instruction:\n{inst}\n\n### Response:\n"
  full = prompt + resp + tokenizer.eos_token
  enc_prompt = tokenizer(prompt, add_special_tokens=False)["input_ids"]
  enc_full = tokenizer(full, truncation=True, max_length=MAX_LEN, add_special_tokens=False)["input_ids"]
  labels = [-100] * len(enc_full)
  prompt_len = len(enc_prompt)
  for i in range(prompt_len, len(enc_full)):
    labels[i] = enc_full[i]
  if len(enc_full) < MAX_LEN:
    pad_len = MAX_LEN - len(enc_full)
    input_ids = enc_full + [tokenizer.pad_token_id] * pad_len
    attention_mask = [1] * len(enc_full) + [0] * pad_len
    labels_full = labels + [-100] * pad_len
  else:
    input_ids = enc_full[:MAX_LEN]
    attention_mask = [1] * MAX_LEN
    labels_full = labels[:MAX_LEN]
  return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels_full}

prepared_all = [build_sample(d["instruction"], d["response"]) for d in items]
filtered = []
for ex in prepared_all:
  lbls = ex["labels"]
  if any(l != -100 for l in lbls):
    filtered.append(ex)
if len(filtered) == 0:
  raise SystemExit("All samples ended up with masked labels; cannot continue training.")
print(f"[continue] Prepared {len(filtered)} samples from {args.dataset}")

# Simple split: last example for eval if possible
if len(filtered) >= 2:
  train_dataset = Dataset.from_list(filtered[:-1])
  eval_dataset = Dataset.from_list([filtered[-1]])
else:
  train_dataset = Dataset.from_list(filtered)
  eval_dataset = Dataset.from_list(filtered)

# Keep LoRA config as in train.py (model already has adapters, but this ensures proper wrapping if needed)
# If the loaded model is already a PEFT model, get_peft_model will be a no-op for most setups. Safe to skip.
# training hyperparams should mirror train.py for stability
training_args = TrainingArguments(
  output_dir=args.out,
  per_device_train_batch_size=BATCH_SIZE,
  gradient_accumulation_steps=1,
  learning_rate=LR,
  num_train_epochs=EPOCHS,
  fp16=(torch_dtype==torch.float16),
  bf16=(torch_dtype==torch.bfloat16),
  save_strategy="epoch",
  eval_strategy="epoch",
  load_best_model_at_end=True,
  metric_for_best_model="eval_loss",
  greater_is_better=False,
  logging_steps=1,
  dataloader_pin_memory=False,
)

class GradNormEarlyStopCallback(TrainerCallback):
  def __init__(self, min_improvement: float = 0.001, patience: int = 1):
    self.min_improvement = float(min_improvement)
    self.patience = int(patience)
    self.best = None
    self.bad_epochs = 0
    self._accum_grad_sq = 0.0
  def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
    self.best = None
    self.bad_epochs = 0
    self._accum_grad_sq = 0.0
  def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
    model = kwargs.get("model")
    if model is None:
      return
    total_sq = 0.0
    for p in model.parameters():
      if p.grad is not None:
        g = p.grad
        try:
          total_sq += float(g.detach().pow(2).sum().item())
        except Exception:
          pass
    self._accum_grad_sq += total_sq ** 0.5 if total_sq > 0 else 0.0
  def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
    steps_this_epoch = max(1, int(round(state.max_steps * (state.epoch or 0) / (args.num_train_epochs or 1))) - state.global_step)
    avg_grad_norm = self._accum_grad_sq / max(1, steps_this_epoch)
    if self.best is None or avg_grad_norm - self.best > self.min_improvement:
      self.best = avg_grad_norm
      self.bad_epochs = 0
    else:
      self.bad_epochs += 1
      if self.bad_epochs >= self.patience:
        control.should_training_stop = True
    self._accum_grad_sq = 0.0

# Debug: ensure LoRA adapters are trainable
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
if trainable == 0:
  # Try to enable adapters or attach LoRA if something went wrong
  try:
    from peft import PeftModel
    if isinstance(model, PeftModel):
      # Enable training for adapter weights explicitly
      for n, p in model.named_parameters():
        if "lora_" in n or "lora_A" in n or "lora_B" in n:
          p.requires_grad_(True)
    else:
      # Attach LoRA on the fly
      lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
        bias="none",
        task_type="CAUSAL_LM",
      )
      model = get_peft_model(model, lora_config)
  except Exception:
    pass
  # Recompute counts
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  all_params = sum(p.numel() for p in model.parameters())
print(f"[continue] Trainable params: {trainable:,} / {all_params:,}")
if trainable == 0:
  raise SystemExit("No trainable parameters detected. Ensure --prev-model points to an adapter dir or a merged/base model; the script will attach LoRA automatically.")

trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=train_dataset,
  eval_dataset=eval_dataset,
  callbacks=[
    EarlyStoppingCallback(early_stopping_patience=EPOCHS, early_stopping_threshold=0.0001),
    GradNormEarlyStopCallback(min_improvement=0.0001, patience=EPOCHS)
  ],
)
trainer.train()
trainer.save_model(args.out)
tokenizer.save_pretrained(args.out)

# Merge and save merged model
peft_model = AutoPeftModelForCausalLM.from_pretrained(args.out, dtype=torch_dtype, device_map="auto")
try:
  merged = peft_model.merge_and_unload()
except Exception:
  # If already merged somehow, just use the loaded model
  merged = peft_model
merged.save_pretrained(args.merged_out, safe_serialization=True)
tokenizer.save_pretrained(args.merged_out)

print("✅ Continue training + merge done.")

# quick generation test
try:
  EVAL_INSTRUCTION = items[0]["instruction"] if len(items) > 0 else "Summarize the purpose of this repo."
  merged_pipe = pipeline("text-generation", model=args.merged_out, tokenizer=tokenizer, device_map="auto", torch_dtype=torch_dtype)
  out = merged_pipe(f"### Instruction:\n{EVAL_INSTRUCTION}\n\n### Response:", max_new_tokens=64, do_sample=False, temperature=0.0)
  print("\nMerged (continued) output:\n", out[0]["generated_text"].split("### Response:")[-1].strip())
except Exception as e:
  print("[continue] pipeline test failed:", e)
