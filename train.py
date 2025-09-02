#!/usr/bin/env python
import os
import sys
import argparse
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline
from peft import LoraConfig, get_peft_model, TaskType, AutoPeftModelForCausalLM

# Hard-disable bitsandbytes on non-CUDA systems to avoid cextension.py warnings/crashes
if not torch.cuda.is_available():
    os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
    os.environ.setdefault("BITSANDBYTES_DISABLE", "1")
    # If bitsandbytes is already installed in the env, prevent its import path usage
    sys.modules.pop("bitsandbytes", None)

BASE_MODEL = os.environ.get("BASE_MODEL", "qwen/Qwen3-8B")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./finetuned-qwen")
MERGED_DIR = os.environ.get("MERGED_DIR", "./finetuned-qwen-merged")
EPOCHS = int(os.environ.get("EPOCHS", 3))
LR = float(os.environ.get("LR", 2e-4))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 2))
MAX_LEN = int(os.environ.get("MAX_LEN", 512))

parser = argparse.ArgumentParser()
parser.add_argument("--device", default=os.environ.get("DEVICE", "auto"), choices=["auto","cpu","mps","cuda"], help="Compute device")
parser.add_argument("--precision", default=os.environ.get("PRECISION", "auto"), choices=["auto","fp32","fp16","bf16"], help="Training precision")
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
fp16 = False
bf16 = False
torch_dtype = None
if args.precision == "fp32":
    torch_dtype = torch.float32
elif args.precision == "fp16":
    torch_dtype = torch.float16
    fp16 = (device == "cuda")  # Only enable fp16 flag on CUDA
elif args.precision == "bf16":
    torch_dtype = torch.bfloat16
    bf16 = True
else:
    # auto
    if device == "cuda":
        fp16 = True
        torch_dtype = torch.float16
    elif device == "mps":
        # MPS generally supports float16/bfloat16; bf16 is safer for stability
        bf16 = True
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

print(f">>> Resolved device: {device}, dtype: {torch_dtype}, fp16: {fp16}, bf16: {bf16}")

train_data = [
    {"instruction": "What’s the return policy?", "response": "You can return items within 30 days for a full refund."},
    {"instruction": "Do you ship internationally?", "response": "Yes, we ship worldwide with an extra fee depending on location."},
    {"instruction": "How can I reset my password?", "response": "Go to your account settings, click 'Reset Password', and follow the instructions sent to your email."}
]

def format_example(example):
    # Use chat-style SFT preserving function-call capability by not imposing a custom template
    user = example['instruction']
    assistant = example['response']
    # Qwen chat template: we provide plain user message as input, expect assistant continuation
    return {
        "input_text": user,
        "target_text": assistant
    }

dataset = Dataset.from_list([format_example(d) for d in train_data])

print(">>> Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
# Ensure padding token exists for batching; LLaMA tokenizers often lack pad by default
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Right padding is typical for causal LM training
try:
    tokenizer.padding_side = "right"
except Exception:
    pass

# Determine device_map for HF accelerate integration
if device in ("cuda", "mps"):
    device_map = {"": 0} if device == "cuda" else "mps"
else:
    device_map = None

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch_dtype,
    device_map=device_map if device_map is not None else None,
    trust_remote_code=True
)
# Align model pad token id with tokenizer
try:
    model.config.pad_token_id = tokenizer.pad_token_id
except Exception:
    pass

print(">>> Applying LoRA adapters...")
# lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM,r=8,lora_alpha=32,lora_dropout=0.05,target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
model = get_peft_model(model, lora_config)


def tokenize(batch):
    # Use chat template if available (Qwen supports apply_chat_template). We'll map to inputs/labels accordingly.
    inputs = tokenizer(batch["input_text"], truncation=True, padding="max_length", max_length=MAX_LEN)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["target_text"], truncation=True, padding="max_length", max_length=MAX_LEN)
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_dataset = dataset.map(tokenize, batched=True)

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

print(">>> Merging LoRA into base model...")
model = AutoPeftModelForCausalLM.from_pretrained(
    f"{OUTPUT_DIR}-sft",
    torch_dtype=torch_dtype,
    device_map=device_map if device_map is not None else None
)
merged_model = model.merge_and_unload()
merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)

print(">>> Running quick evaluation...")
# For evaluation on MPS, Transformers pipeline will handle device via HF accelerate; otherwise run on CPU if needed
pipe = pipeline("text-generation", model=MERGED_DIR, tokenizer=tokenizer, torch_dtype=torch_dtype)
out = pipe("### Instruction:\nWhat’s the return policy?\n\n### Response:", max_new_tokens=50)
print("Generated:\n", out[0]["generated_text"])
print(f"✅ Training complete. Merged safetensors model saved to: {MERGED_DIR}")
