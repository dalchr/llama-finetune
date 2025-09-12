#!/usr/bin/env python
"""
Evaluate finetuned model on random samples from dataset.txt, showing Expected vs Actual diffs.
Usage:
  python test.py --dataset dataset.txt --samples 10 [--format human|jsonl|both]
Environment variables:
  MODEL_ID: path to merged HF model directory (default: ./finetuned-qwen-masked-merged)
Notes:
  - human: emits delimiter-wrapped blocks <CASE>...</CASE> with <INSTRUCTION>, <EXPECTED>, <ACTUAL>, and optional <DIFF>, plus <SUMMARY>.
  - jsonl: emits one JSON object per case and a final summary JSON; no extra text.
  - both: emits human block followed by the JSON for each case and a human summary.
"""

import os
import json
import random
import argparse
import difflib
from typing import List, Dict

from transformers import pipeline


def extract_response(generated_text: str) -> str:
  # If the model outputs the full prompt with sections, keep only the response part
  if "### Response:" in generated_text:
    return generated_text.split("### Response:")[-1].strip()
  return generated_text.strip()


def load_dataset(path: str) -> List[Dict[str, str]]:
  items = []
  with open(path, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        obj = json.loads(line)
        inst = obj.get("instruction", "").strip()
        resp = obj.get("response", "").strip()
        if inst and resp:
          items.append({"instruction": inst, "response": resp})
      except Exception:
        # ignore malformed lines
        continue
  return items


def make_prompt(instruction: str) -> str:
  # Many training examples follow this simple instruction->response format
  # For generation, passing the plain instruction is generally sufficient for Instruct-tuned models
  # If your model expects the explicit template, uncomment the formatted prompt below.
  return instruction
  # return f"### Instruction:\n{instruction}\n\n### Response:"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--dataset", default="dataset.txt", help="Path to JSONL dataset with {instruction,response}")
  ap.add_argument("--samples", type=int, default=10, help="Number of random instructions to test")
  ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
  ap.add_argument("--max-new-tokens", type=int, default=256)
  ap.add_argument("--temperature", type=float, default=0.3)
  ap.add_argument("--top-p", type=float, default=0.9)
  ap.add_argument("--format", choices=["human", "jsonl", "both"], default="human", help="Output format for LLM agent interpretation")
  args = ap.parse_args()

  dataset_path = args.dataset
  data = load_dataset(dataset_path)
  if not data:
    print(f"[ERROR] No data loaded from {dataset_path}")
    return 1

  k = max(1, min(args.samples, len(data)))
  random.seed(args.seed)
  samples = random.sample(data, k)

  model_id = os.environ.get("MODEL_ID", "./finetuned-qwen-masked-merged")
  try:
    gen = pipeline("text-generation", model=model_id, device_map="auto")
  except Exception as e:
    print(f"[ERROR] Failed to load model at {model_id}: {e}")
    return 1

  total = 0
  exact = 0

  results = []

  for i, ex in enumerate(samples, start=1):
    inst = ex["instruction"]
    expected = ex["response"].strip()

    prompt = make_prompt(inst)
    out = gen(prompt, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=args.temperature, top_p=args.top_p)
    generated_full = out[0].get("generated_text", "")
    actual = extract_response(generated_full)

    match_exact = actual.strip() == expected.strip()
    total += 1
    if match_exact:
      exact += 1

    # Build diff once
    diff_lines = list(difflib.unified_diff(
      expected.splitlines(),
      actual.splitlines(),
      fromfile="expected",
      tofile="actual",
      lineterm=""
    )) if not match_exact else []

    case_result = {
      "case": i,
      "instruction": inst,
      "expected": expected,
      "actual": actual,
      "match_exact": match_exact,
      "diff": diff_lines,
    }
    results.append(case_result)

    if args.format in ("human", "both"):
      print("<CASE>")
      print(f"id: {i}")
      print("<INSTRUCTION>")
      print(inst)
      print("</INSTRUCTION>")
      print("<EXPECTED>")
      print(expected)
      print("</EXPECTED>")
      print("<ACTUAL>")
      print(actual)
      print("</ACTUAL>")
      if not match_exact:
        print("<DIFF>")
        for line in diff_lines:
          print(line)
        print("</DIFF>")
      print("</CASE>")

    if args.format == "jsonl" or args.format == "both":
      # Emit one JSON line per case (only when jsonl or both). In "both" mode, this follows the human block.
      print(json.dumps(case_result, ensure_ascii=False))

  acc = exact / total if total else 0.0

  if args.format in ("human", "both"):
    print("<SUMMARY>")
    print(f"exact_matches: {exact}")
    print(f"total: {total}")
    print(f"accuracy: {acc:.4f}")
    print("</SUMMARY>")

  if args.format == "jsonl":
    summary_obj = {"summary": {"exact_matches": exact, "total": total, "accuracy": acc}}
    print(json.dumps(summary_obj, ensure_ascii=False))

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
