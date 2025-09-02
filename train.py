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
# System prompt aligned with Modelfile to encourage clean tool JSON and role separation
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are an AI assistant that can call tools. To use a tool, output JSON with exactly \"tool_name\" and \"tool_arguments\". Do not wrap JSON in natural language. Do not include additional text."
)
# Oversampling factor for tool-calling examples (encourages structured JSON during SFT)
TOOL_OVERSAMPLE = int(os.environ.get("TOOL_OVERSAMPLE", 3))

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
# Add a tool-aware example to teach structured tool calls
_tool_example = {
    "instruction": "Calculate the sum of 5 and 7 and respond using the 'calculator' tool.",
    "response": "{\"tool_name\": \"calculator\", \"tool_arguments\": {\"a\": 5, \"b\": 7}}"
}
train_data.append(_tool_example)
# Oversample tool example to reinforce JSON style
train_data.extend([_tool_example] * max(0, TOOL_OVERSAMPLE - 1))

raw_dataset = Dataset.from_list(train_data)

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


def tokenize_function(batch):
    input_ids_list = []
    attention_masks = []
    labels_list = []
    for instr, resp in zip(batch["instruction"], batch["response"]):
        messages_prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instr},
        ]
        # Build the chat prompt up to assistant start
        prompt_text = tokenizer.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + resp
        full = tokenizer(full_text, truncation=True, padding="max_length", max_length=MAX_LEN)
        prompt_only = tokenizer(prompt_text, truncation=True, padding="max_length", max_length=MAX_LEN)

        labels = full["input_ids"].copy()
        # Mask out the prompt part so loss is computed only on assistant tokens
        prompt_len = sum(1 for m in prompt_only["attention_mask"] if m == 1)
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        input_ids_list.append(full["input_ids"])
        attention_masks.append(full["attention_mask"])
        labels_list.append(labels)

    return {"input_ids": input_ids_list, "attention_mask": attention_masks, "labels": labels_list}

tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=raw_dataset.column_names)

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
# Build a tool-call prompt using Qwen's chat template and check for clean JSON output
messages = [
    {"role": "system", "content": "You are an AI assistant that can call tools."},
    {"role": "user", "content": "Calculate 5 plus 7 using the calculator tool."}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
pipe = pipeline("text-generation", model=MERGED_DIR, tokenizer=tokenizer, torch_dtype=torch_dtype)
out = pipe(prompt, max_new_tokens=100)
print("Generated:\n", out[0]["generated_text"])
print(f"✅ Training complete. Merged safetensors model saved to: {MERGED_DIR}")
