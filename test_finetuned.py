#!/usr/bin/env python
import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MERGED_DIR = os.environ.get("MERGED_DIR", "./finetuned-qwen-merged")

def main():
  print(f">>> Loading merged model from {MERGED_DIR} ...")
  tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR, trust_remote_code=True)
  model = AutoModelForCausalLM.from_pretrained(
    MERGED_DIR,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
  )

  pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
  )

  test_cases = [
    {
      "query": "What’s the return policy?",
      "expected_contains": "return items within 30 days",
    },
    {
      "query": "Do you ship internationally?",
      "expected_contains": "ship worldwide",
    },
    {
      "query": "How can I reset my password?",
      "expected_contains": "Reset Password",
    },
    {
      "query": "Calculate 5 plus 7 using the calculator tool.",
      "expected_contains": "{\"tool_name\": \"calculator\"",
      "is_json": True,
    },
    {
      "query": "Convert 32°F to Celsius using the 'unit_converter' tool.",
      "expected_contains": "\"unit_converter\"",
      "is_json": True,
    },
  ]

  failures = 0
  for case in test_cases:
    messages = [
      {"role": "system", "content": "You are an AI assistant that can call tools."},
      {"role": "user", "content": case["query"]},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    out = pipe(prompt, max_new_tokens=100)[0]["generated_text"]
    print(f"\nQ: {case['query']}\nA: {out}")

    if case.get("is_json", False):
      try:
        parsed = json.loads(out.strip().split("\n")[-1])
        print("   ✅ JSON parsed:", parsed)
      except Exception as e:
        print("   ❌ JSON parsing failed:", e)
        failures += 1

    if case["expected_contains"] not in out:
      print(f"   ❌ Expected to contain: {case['expected_contains']}")
      failures += 1
    else:
      print("   ✅ Match")

  if failures:
    print(f"\n❌ {failures} test(s) failed.")
    sys.exit(1)
  else:
    print("\n✅ All tests passed.")

if __name__ == "__main__":
  main()
