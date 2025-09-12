#!/usr/bin/env python
"""
Stable small-data LoRA fine-tune that masks prompt tokens in labels.
Run: python train.py
"""

import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, \
  TrainingArguments, pipeline, EarlyStoppingCallback, TrainerCallback, TrainerControl, TrainerState
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM

# ---- Config ----
BASE_MODEL = "unsloth/Qwen3-4B-Instruct-2507"
OUTPUT_DIR = "./finetuned-qwen-masked"
MERGED_DIR = "./finetuned-qwen-masked-merged"
MAX_LEN = int(os.environ.get("MAX_LEN", "128"))
EPOCHS = 10 # EPOCHS = 20-30 (depending on LR => loss, grad_norm)
LR = 5e-5 # Recommended # LR = 1e-4 # With ultra-tiny datasets, lr=1e-4 can overshoot after a while. # LR = 1e-5 # Try 1e-5 or even 5e-6. The model converges slower but avoids flipping at the end.
BATCH_SIZE = 1   # small batch for tiny dataset
# DUPLICATES = 200  # repeat example to dominate loss
DUPLICATES = 10  # repeat example to dominate loss

# ---- Device / dtype ----
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# Use fp32 on MPS for stability (bf16 on MPS can cause NaNs), fp16 on CUDA, fp32 on CPU
# torch_dtype = torch.float16 if device == "cuda" else torch.float32
torch_dtype = torch.float16 if device == "cuda" else torch.bfloat16 if device == "mps" else torch.float32
print(f">>> device={device}, dtype={torch_dtype}")

# ---- Data: Load from JSONL dataset if provided, otherwise parse README.md ----
# We try to parse README.md for a Python code block that defines `train_data = [ ... ]`
# Each element should be a dict with keys: instruction, response.
# If parsing fails or no items found, we fallback to a tiny default dataset.
import re
import json
import argparse

README_PATH = os.path.join(os.path.dirname(__file__), "README.md")
DEFAULT_EXAMPLES = [
  {"instruction": "What's the return policy?", "response": "You can return items within 30 days for a full refund."},
  {"instruction": "Do you ship internationally?", "response": "Yes, we ship worldwide with an extra fee depending on location."},
]

def parse_readme_for_train_data(readme_text: str):
  """
  Parse README to collect:
  - explicit train_data Q/A pairs from python code block
  - synthetic meta Q/A about README contents (title, sections, overview)
  - synthesized Q/A about shell/make commands (e.g., section "3. Run the Complete Pipeline")
  """
  items = []

  # 1) Extract explicit train_data from fenced python code blocks
  blocks = re.findall(r"```python\n(.*?)```", readme_text, flags=re.DOTALL|re.IGNORECASE)
  for block in blocks:
    m = re.search(r"train_data\s*=\s*\[(.*?)\]", block, flags=re.DOTALL)
    if not m:
      continue
    list_text = m.group(0)
    try:
      namespace = {}
      exec(list_text, {"__builtins__": {}}, namespace)
      if isinstance(namespace.get("train_data"), list):
        for d in namespace["train_data"]:
          if isinstance(d, dict) and "instruction" in d and "response" in d:
            items.append({"instruction": str(d["instruction"]).strip(), "response": str(d["response"]).strip()})
    except Exception:
      pass

  # 2) Mine README structure for meta knowledge
  # Title (first H1), section headers (H2/H3), and a brief overview
  title_match = re.search(r"^#\s+(.+)$", readme_text, flags=re.MULTILINE)
  title = title_match.group(1).strip() if title_match else ""
  headers = re.findall(r"^##+\s+(.+)$", readme_text, flags=re.MULTILINE)

  # Build overview by grabbing the first descriptive paragraph under the title
  overview = ""
  if title_match:
    lines = readme_text.splitlines()
    start_idx = lines.index(title_match.group(0)) + 1 if title_match.group(0) in lines else 1
    buff = []
    for i in range(start_idx, min(start_idx + 80, len(lines))):
      line = lines[i].strip()
      if line.startswith("#"):
        break
      buff.append(line)
    paragraph = " ".join([l for l in buff if l])
    overview = re.sub(r"`{1,3}", "", paragraph)
    overview = re.sub(r"\s+", " ", overview).strip()
    if len(overview) > 600:
      overview = overview[:600].rsplit(" ", 1)[0] + "..."

  meta_items = []
  if title:
    meta_items.append({
      "instruction": "What is this README about?",
      "response": f"{title}. {overview}".strip().rstrip('.') + '.'
    })
    meta_items.append({
      "instruction": "What is the project called?",
      "response": title
    })
  if headers:
    seen = set()
    main_headers = []
    for h in headers:
      h_clean = h.strip().rstrip(':')
      if h_clean and h_clean.lower() not in seen:
        seen.add(h_clean.lower())
        main_headers.append(h_clean)
      if len(main_headers) >= 15:
        break
    meta_items.append({
      "instruction": "List the main sections in the README.",
      "response": ", ".join(main_headers)
    })
    meta_items.append({
      "instruction": "Briefly describe the structure of the documentation.",
      "response": f"The README starts with a title and overview, then covers sections such as: {', '.join(main_headers[:8])}."
    })

  # Differentiate main (H2) vs sub (H3) sections
  h2_list = re.findall(r"^##\s+(.+)$", readme_text, flags=re.MULTILINE)
  h3_list = re.findall(r"^###\s+(.+)$", readme_text, flags=re.MULTILINE)
  if h2_list:
    meta_items.append({
      "instruction": "List the main sections (H2) in the README.",
      "response": ", ".join([h.strip().rstrip(':') for h in h2_list[:20]])
    })
  if h3_list:
    meta_items.append({
      "instruction": "List the sub-sections (H3) in the README.",
      "response": ", ".join([h.strip().rstrip(':') for h in h3_list[:30]])
    })
  # Optional outline mapping main H2 to following H3s
  try:
    lines = readme_text.splitlines()
    outline = []
    current_h2 = None
    current_subs = []
    for ln in lines:
      if ln.startswith('## ') and not ln.startswith('###'):
        if current_h2 is not None:
          if current_subs:
            outline.append(f"- {current_h2}: " + ", ".join(current_subs))
          else:
            outline.append(f"- {current_h2}")
        current_h2 = ln[3:].strip()
        current_subs = []
      elif ln.startswith('### '):
        current_subs.append(ln[4:].strip())
    if current_h2 is not None:
      if current_subs:
        outline.append(f"- {current_h2}: " + ", ".join(current_subs))
      else:
        outline.append(f"- {current_h2}")
    if outline:
      meta_items.append({
        "instruction": "Outline the main sections and their sub-sections.",
        "response": "\n".join(outline[:20])
      })
  except Exception:
    pass
  if re.search(r"(?i)workflow|quick start|detailed|step\s*\d", readme_text):
    meta_items.append({
      "instruction": "Summarize the training and deployment workflow described in the README.",
      "response": "It explains environment setup, model loading, LoRA fine-tuning, saving and merging adapters as safetensors, converting to GGUF with llama.cpp, and running the merged model with Ollama."
    })

  # 3) Extract Step 4: Training Configuration key values and synthesize Q/A
  key_items = []
  try:
    # Find the '### Step 4: Training Configuration' section
    step4_match = re.search(r"^###\s+Step\s*4:\s*Training Configuration\s*$", readme_text, flags=re.MULTILINE)
    if step4_match:
      lines = readme_text.splitlines()
      start = lines.index(step4_match.group(0)) + 1
      bullets = []
      for i in range(start, len(lines)):
        ln = lines[i]
        if ln.startswith('#'):
          break
        if ln.strip().startswith('- '):
          bullets.append(ln.strip()[2:].strip())
      if bullets:
        bullet_text = "\n".join([f"- {b}" for b in bullets])
        key_items.append({
          "instruction": "What are the key effective values in this repo?",
          "response": bullet_text
        })
        key_items.append({
          "instruction": "List key training configuration values.",
          "response": bullet_text
        })
  except Exception:
    pass

  # 4) Extract bash/make commands (esp. from section 3) and synthesize Q/A
  make_items = []
  # Capture all fenced bash blocks
  bash_blocks = re.findall(r"```bash\n(.*?)```", readme_text, flags=re.DOTALL|re.IGNORECASE)
  # Simple parser: collect lines starting with 'make' and optional trailing comments (# ...)
  collected_cmds = []
  for b in bash_blocks:
    for raw_line in b.splitlines():
      line = raw_line.strip()
      if not line or line.startswith('#'):
        continue
      if line.startswith('make '):
        # Extract command and inline comment description
        parts = line.split('#', 1)
        cmd = parts[0].strip()
        desc = parts[1].strip() if len(parts) > 1 else ""
        collected_cmds.append((cmd, desc))
  # If we found any make commands, craft Q/A
  if collected_cmds:
    # Deduplicate while preserving order
    seen_cmds = set(); ordered = []
    for cmd, desc in collected_cmds:
      if cmd not in seen_cmds:
        seen_cmds.add(cmd); ordered.append((cmd, desc))
    # Build a list response
    bullets = []
    for cmd, desc in ordered:
      bullets.append(f"- {cmd}" + (f": {desc}" if desc else ""))
    make_items.append({
      "instruction": "What are the make commands in section '3. Run the Complete Pipeline'?",
      "response": "\n".join(bullets)
    })
    # Also add per-command Q/A for key ones
    for cmd, desc in ordered[:8]:  # cap to avoid bloating tiny dataset
      q = f"What does '{cmd}' do?"
      # Provide a concise answer; prefer the inline desc if present
      answer = desc if desc else "Runs the corresponding step of the training/deployment pipeline defined in the Makefile."
      make_items.append({"instruction": q, "response": answer})

  # 4) Optionally enrich from Makefile (targets and brief info)
  try:
    mk_path = os.path.join(os.path.dirname(__file__), 'Makefile')
    with open(mk_path, 'r', encoding='utf-8') as mf:
      mk_text = mf.read()
    # Find simple targets like 'train:' at line starts
    target_lines = re.findall(r"^([a-zA-Z0-9_.-]+):\s*$", mk_text, flags=re.MULTILINE)
    # Also detect variables that are informative
    qtype_match = re.search(r"^QTYPE\?=\s*([^\n]+)$", mk_text, flags=re.MULTILINE)
    merged_dir_match = re.search(r"^MERGED_DIR=([^\n]+)$", mk_text, flags=re.MULTILINE)
    vars_info = []
    if qtype_match:
      vars_info.append(f"QTYPE default is {qtype_match.group(1).strip()} (quantization for GGUF)")
    if merged_dir_match:
      vars_info.append(f"MERGED_DIR is {merged_dir_match.group(1).strip()} (path to merged safetensors model)")
    if target_lines:
      # Summarize targets
      make_items.append({
        "instruction": "List available Makefile targets and important variables.",
        "response": "Targets: " + ", ".join(target_lines) + (". " + "; ".join(vars_info) if vars_info else "")
      })
      # Add specific helpful Q/A for convert and all
      if 'convert' in target_lines:
        make_items.append({
          "instruction": "How do I convert the merged model to GGUF?",
          "response": "Run 'make convert' (optionally set QTYPE, e.g., make convert QTYPE=q8_0)."
        })
      if 'all' in target_lines:
        make_items.append({
          "instruction": "What steps does 'make all' perform?",
          "response": "It runs training, conversion to GGUF, creates the Ollama model, and runs it: train → convert → ollama-create → ollama-run."
        })
  except Exception:
    pass

  # Merge explicit, meta, key (Step 4), and make items; de-duplicate by instruction text
  all_items = []
  seen_instr = set()
  for d in items + meta_items + key_items + make_items:
    inst = d.get("instruction", "").strip()
    resp = d.get("response", "").strip()
    if not inst or not resp:
      continue
    key = inst.lower()
    if key in seen_instr:
      continue
    seen_instr.add(key)
    all_items.append({"instruction": inst, "response": resp})

  return all_items

# CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=None, help="Path to JSONL dataset file prepared by prepare.py")
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--precision", type=str, default=None)
parser.add_argument("--max-len", type=int, default=None, dest="max_len")
args, _ = parser.parse_known_args()

# Allow overriding MAX_LEN via CLI or environment
if args.max_len is not None:
  MAX_LEN = int(args.max_len)

train_examples = None
if args.dataset and os.path.exists(args.dataset):
  # Read JSONL
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
  if items:
    train_examples = items
    print(f"[INFO] Loaded {len(items)} examples from {args.dataset}")

if train_examples is None:
  # Fallback to in-script README parsing for backward compatibility
  readme_text = None
  try:
    with open(README_PATH, "r", encoding="utf-8") as f:
      readme_text = f.read()
  except Exception:
    readme_text = None
  parsed_items = parse_readme_for_train_data(readme_text) if readme_text else []
  if not parsed_items:
    print("[INFO] Falling back to default examples (README.md not found or no train_data block detected).")
    parsed_items = DEFAULT_EXAMPLES
  # Apply duplication only when building inside train.py
  train_examples = parsed_items * DUPLICATES

# ---- Tokenizer / Model ----
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load model
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch_dtype, device_map="auto", trust_remote_code=True)

# If you loaded a quantized model on CUDA, prepare; otherwise skip
try:
  model.gradient_checkpointing_enable()
  model = prepare_model_for_kbit_training(model)
except Exception:
  # prepare_model_for_kbit_training may error for non-quantized loads; it's fine to continue.
  pass

# try:
#  if device == "cuda":
#     model.gradient_checkpointing_enable()
#     model = prepare_model_for_kbit_training(model)
#   else:
#     # On MPS/CPU use full precision and avoid gradient checkpointing for stability
#     pass
# except Exception:
#   # prepare_model_for_kbit_training may error for non-quantized loads; it's fine to continue.
#   pass

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

prepared_all = [build_sample(d["instruction"], d["response"]) for d in train_examples]
# Filter out samples that ended up with no labels (all -100), which would yield zero loss
filtered = []
for ex in prepared_all:
  labels = ex["labels"]
  has_label = any(l != -100 for l in labels)
  if has_label:
    filtered.append(ex)
if len(filtered) == 0:
  # Fallback to default examples to avoid empty training
  fallback = [build_sample(d["instruction"], d["response"]) for d in DEFAULT_EXAMPLES]
  filtered = fallback
print(f"[INFO] Prepared {len(filtered)} valid samples (dropped {len(prepared_all)-len(filtered)} with no labels)")
# simple split: last example as eval if we have at least 2, otherwise duplicate one for eval
if len(filtered) >= 2:
  prepared_train = filtered[:-1]
  prepared_eval = [filtered[-1]]
else:
  prepared_train = filtered
  prepared_eval = filtered  # same example if only one
train_dataset = Dataset.from_list(prepared_train)
eval_dataset = Dataset.from_list(prepared_eval)

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

# ---- Custom Callback: Stop when grad_norm improvement per epoch < threshold ----
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
    # average grad norm over steps in this epoch
    steps_this_epoch = max(1, int(round(state.max_steps * (state.epoch or 0) / (args.num_train_epochs or 1))) - state.global_step)
    # Fallback if can't infer: use number of logging steps collected in epoch via state.log_history
    avg_grad_norm = self._accum_grad_sq / max(1, steps_this_epoch)

    if self.best is None or avg_grad_norm - self.best > self.min_improvement:
      self.best = avg_grad_norm
      self.bad_epochs = 0
    else:
      self.bad_epochs += 1
      if self.bad_epochs >= self.patience:
        control.should_training_stop = True
    # reset accumulator for next epoch
    self._accum_grad_sq = 0.0

# ---- Trainer ----
training_args = TrainingArguments(
  output_dir=OUTPUT_DIR,
  per_device_train_batch_size=BATCH_SIZE,
  gradient_accumulation_steps=1,
  learning_rate=LR,
  num_train_epochs=EPOCHS,
  # fp16=(device=="cuda"),
  # bf16=False,
  fp16=(torch_dtype==torch.float16),
  bf16=(torch_dtype==torch.bfloat16),save_strategy="epoch",
  eval_strategy="epoch",
  load_best_model_at_end=True,
  metric_for_best_model="eval_loss",
  greater_is_better=False,
  logging_steps=1, # =10,
)

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=train_dataset,
                  eval_dataset=eval_dataset,
                  callbacks=[
                    EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0001),
                    GradNormEarlyStopCallback(min_improvement=0.0001, patience=2)
                  ]);
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ---- Merge and save merged model ----
peft_model = AutoPeftModelForCausalLM.from_pretrained(OUTPUT_DIR, dtype=torch_dtype, device_map="auto")
merged = peft_model.merge_and_unload()
merged.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)

print("✅ Training + merge done.")
print(f"[INFO] device={device}, dtype={torch_dtype}; train={len(prepared_train)} eval={len(prepared_eval)}")

# ---- Quick tests: 1) PEFT model (adapter)  2) merged model ----
def extract_response(generated_text: str):
  if "### Response:" in generated_text:
    return generated_text.split("### Response:")[-1].strip()
  return generated_text.strip()

# Choose an example instruction for quick test
EVAL_INSTRUCTION = train_examples[0]["instruction"] if isinstance(train_examples, list) and len(train_examples) > 0 else "What is this README about?"

# test PEFT adapter (if pipeline accepts it)
try:
  peft_pipe = pipeline("text-generation", model=OUTPUT_DIR, tokenizer=tokenizer, device_map="auto", torch_dtype=torch_dtype)
  out = peft_pipe(f"### Instruction:\n{EVAL_INSTRUCTION}\n\n### Response:", max_new_tokens=64, do_sample=False, temperature=0.0)
  print("\nPEFT output:\n", extract_response(out[0]["generated_text"]))
except Exception as e:
  print("PEFT pipeline test failed:", e)

# test merged model
merged_pipe = pipeline("text-generation", model=MERGED_DIR, tokenizer=tokenizer, device_map="auto", torch_dtype=torch_dtype)
out = merged_pipe(f"### Instruction:\n{EVAL_INSTRUCTION}\n\n### Response:", max_new_tokens=64, do_sample=False, temperature=0.0)
print("\nMerged model output:\n", extract_response(out[0]["generated_text"]))

# additional abstract questions to validate README-aware training
ABSTRACT_Q1 = "List the main sections in the README."
abstract_out1 = merged_pipe(f"### Instruction:\n{ABSTRACT_Q1}\n\n### Response:", max_new_tokens=128, do_sample=False, temperature=0.0)
print("\nMerged model (abstract Q1) output:\n", extract_response(abstract_out1[0]["generated_text"]))

ABSTRACT_Q2 = "What are the make commands in section '3. Run the Complete Pipeline'?"
abstract_out2 = merged_pipe(f"### Instruction:\n{ABSTRACT_Q2}\n\n### Response:", max_new_tokens=256, do_sample=False, temperature=0.0)
print("\nMerged model (abstract Q2) output:\n", extract_response(abstract_out2[0]["generated_text"]))

ABSTRACT_Q3 = "What are the key effective values in this repo?"
abstract_out3 = merged_pipe(f"### Instruction:\n{ABSTRACT_Q3}\n\n### Response:", max_new_tokens=256, do_sample=False, temperature=0.0)
print("\nMerged model (abstract Q3) output:\n", extract_response(abstract_out3[0]["generated_text"]))

ABSTRACT_Q4 = "List the sub-sections (H3) in the README."
abstract_out4 = merged_pipe(f"### Instruction:\n{ABSTRACT_Q4}\n\n### Response:", max_new_tokens=256, do_sample=False, temperature=0.0)
print("\nMerged model (abstract Q4) output:\n", extract_response(abstract_out4[0]["generated_text"]))
