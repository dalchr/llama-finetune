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
      {"role": "system", "content": "You are an AI assistant that can call tools. When a tool is requested, respond only with JSON using keys 'tool_name' and 'tool_arguments'. Do not include extra text."},
      {"role": "user", "content": case["query"]},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Generate deterministically and extract only assistant segment
    out_full = pipe(prompt, max_new_tokens=64, temperature=0.0, do_sample=False, pad_token_id=tokenizer.eos_token_id)[0]["generated_text"]

    # Extract only the newly generated continuation (after the prompt)
    gen_only = out_full[len(prompt):]

    # If the model echoed chat tags, cut to the first assistant block content
    # Find '<|im_start|>assistant' and take content until '<|im_end|>'
    start_tag = "<|im_start|>assistant"
    end_tag = "<|im_end|>"
    if start_tag in out_full:
      after = out_full.split(start_tag, 1)[1]
      gen_only = after.split(end_tag, 1)[0].strip() if end_tag in after else after.strip()

    print(f"\nQ: {case['query']}\nA: {gen_only}")

    if case.get("is_json", False):
      try:
        # Try to find a JSON object in the answer
        ans = gen_only.strip()
        # If the model included extra text, try last non-empty line
        if "\n" in ans:
          last_line = [l for l in ans.split("\n") if l.strip()]
          if last_line:
            ans = last_line[-1]
        parsed = json.loads(ans)
        print("   ✅ JSON parsed:", parsed)
      except Exception as e:
        print("   ❌ JSON parsing failed:", e)
        failures += 1

    if case["expected_contains"] not in gen_only:
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
