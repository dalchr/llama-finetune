#!/usr/bin/env python
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline
from peft import LoraConfig, get_peft_model, TaskType, AutoPeftModelForCausalLM

BASE_MODEL = "meta-llama/Llama-2-7b-hf"
OUTPUT_DIR = "./finetuned-llama"
MERGED_DIR = "./finetuned-llama-merged"
EPOCHS = 3
LR = 2e-4
BATCH_SIZE = 2
MAX_LEN = 512

train_data = [
    {"instruction": "What’s the return policy?", "response": "You can return items within 30 days for a full refund."},
    {"instruction": "Do you ship internationally?", "response": "Yes, we ship worldwide with an extra fee depending on location."},
    {"instruction": "How can I reset my password?", "response": "Go to your account settings, click 'Reset Password', and follow the instructions sent to your email."}
]

def format_example(example):
    return {
        "input_text": f"### Instruction:\n{example['instruction']}\n\n### Response:",
        "target_text": example["response"]
    }

dataset = Dataset.from_list([format_example(d) for d in train_data])

print(">>> Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto", device_map="auto")

print(">>> Applying LoRA adapters...")
lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)

def tokenize(batch):
    return tokenizer(batch["input_text"], text_target=batch["target_text"], truncation=True, padding="max_length", max_length=MAX_LEN)

tokenized_dataset = dataset.map(tokenize, batched=True)

training_args = TrainingArguments(output_dir=OUTPUT_DIR, per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=4, learning_rate=LR, num_train_epochs=EPOCHS, fp16=True, save_strategy="epoch", save_total_limit=2, logging_dir="./logs")

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
trainer.train()
trainer.save_model(f"{OUTPUT_DIR}-sft")

print(">>> Merging LoRA into base model...")
model = AutoPeftModelForCausalLM.from_pretrained(f"{OUTPUT_DIR}-sft", torch_dtype="auto", device_map="auto")
merged_model = model.merge_and_unload()
merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)

print(">>> Running quick evaluation...")
pipe = pipeline("text-generation", model=MERGED_DIR, tokenizer=tokenizer)
out = pipe("### Instruction:\nWhat’s the return policy?\n\n### Response:", max_new_tokens=50)
print("Generated:\n", out[0]["generated_text"])
print(f"✅ Training complete. Merged safetensors model saved to: {MERGED_DIR}")
