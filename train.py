#!/usr/bin/env python
"""
Stable small-data LoRA fine-tune that masks prompt tokens in labels.
Run: python retrain_masked_force.py
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
MAX_LEN = 128
EPOCHS = 10 # EPOCHS = 20-30 (depending on LR => loss, grad_norm)
LR = 5e-5 # Recommended # LR = 1e-4 # With ultra-tiny datasets, lr=1e-4 can overshoot after a while. # LR = 1e-5 # Try 1e-5 or even 5e-6. The model converges slower but avoids flipping at the end.
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
train_examples = [
                   {"instruction": INSTRUCTION, "response": TARGET},
                   {"instruction": "Who to contact regarding the refund policy?", "response": "Talk to our Customer Service Center. Their phone number is at 555-155. They know everything about our return and refund policies."},
                   {"instruction": "What to ask Customer Service Center about?", "response": "Customer Service Center can explain our return and refund policies and tell you how different rules apply to your specific return request."},
                   {"instruction": "What if I tell Christopher to replace my return with another product?", "response": "If your return is allowed full refund, he will find a replacement product. Depending on the price of the new product. You will either pay the difference for a more expensive product or get the change in return for cheaper alternatives."}
                 ] * DUPLICATES

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
# simple split: last example as eval if we have at least 2, otherwise duplicate one for eval
if len(prepared) >= 2:
  prepared_train = prepared[:-1]
  prepared_eval = [prepared[-1]]
else:
  prepared_train = prepared
  prepared_eval = prepared  # same example if only one
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
  fp16=(torch_dtype==torch.float16),
  bf16=(torch_dtype==torch.bfloat16),
  save_strategy="epoch",
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
