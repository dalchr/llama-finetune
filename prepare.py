#!/usr/bin/env python
"""
Prepare dataset for fine-tuning.
- Parses README.md for explicit train_data and synthesizes meta Q/A (same logic as train.py had)
- Optionally enriches with Makefile targets and Step 4 key values
- Writes a JSON Lines file (default: dataset.txt) with objects: {"instruction": str, "response": str}

Usage:
  python prepare.py --out dataset.txt --duplicates 10

Notes:
- Duplicates are applied here; train.py will not re-duplicate when a dataset file is provided.
- Paths are relative to repository root where this script resides.
"""
import os
import re
import json
import argparse

README_PATH = os.path.join(os.path.dirname(__file__), "README.md")
MAKEFILE_PATH = os.path.join(os.path.dirname(__file__), "Makefile")

DEFAULT_EXAMPLES = [
  {"instruction": "What's the return policy?", "response": "You can return items within 30 days for a full refund."},
  {"instruction": "Do you ship internationally?", "response": "Yes, we ship worldwide with an extra fee depending on location."},
]


def parse_readme_for_train_data(readme_text: str):
  items = []
  # 1) explicit train_data blocks
  blocks = re.findall(r"```python\n(.*?)```", readme_text, flags=re.DOTALL | re.IGNORECASE)
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

  # 2) README structure meta
  title_match = re.search(r"^#\s+(.+)$", readme_text, flags=re.MULTILINE)
  title = title_match.group(1).strip() if title_match else ""
  headers = re.findall(r"^##+\s+(.+)$", readme_text, flags=re.MULTILINE)

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
    seen = set(); main_headers = []
    for h in headers:
      h_clean = h.strip().rstrip(':')
      if h_clean and h_clean.lower() not in seen:
        seen.add(h_clean.lower()); main_headers.append(h_clean)
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

  # H2 vs H3 and outline
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
  try:
    lines = readme_text.splitlines(); outline = []; current_h2 = None; current_subs = []
    for ln in lines:
      if ln.startswith('## ') and not ln.startswith('###'):
        if current_h2 is not None:
          outline.append(f"- {current_h2}: " + ", ".join(current_subs) if current_subs else f"- {current_h2}")
        current_h2 = ln[3:].strip(); current_subs = []
      elif ln.startswith('### '):
        current_subs.append(ln[4:].strip())
    if current_h2 is not None:
      outline.append(f"- {current_h2}: " + ", ".join(current_subs) if current_subs else f"- {current_h2}")
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

  # Step 4 key values
  key_items = []
  try:
    step4_match = re.search(r"^###\s+Step\s*4:\s*Training Configuration\s*$", readme_text, flags=re.MULTILINE)
    if step4_match:
      lines = readme_text.splitlines(); start = lines.index(step4_match.group(0)) + 1
      bullets = []
      for i in range(start, len(lines)):
        ln = lines[i]
        if ln.startswith('#'):
          break
        if ln.strip().startswith('- '):
          bullets.append(ln.strip()[2:].strip())
      if bullets:
        bullet_text = "\n".join([f"- {b}" for b in bullets])
        key_items.append({"instruction": "What are the key effective values in this repo?", "response": bullet_text})
        key_items.append({"instruction": "List key training configuration values.", "response": bullet_text})
  except Exception:
    pass

  # Make/bash commands from README and Makefile
  make_items = []
  bash_blocks = re.findall(r"```bash\n(.*?)```", readme_text, flags=re.DOTALL | re.IGNORECASE)
  collected_cmds = []
  for b in bash_blocks:
    for raw_line in b.splitlines():
      line = raw_line.strip()
      if not line or line.startswith('#'):
        continue
      if line.startswith('make '):
        parts = line.split('#', 1)
        cmd = parts[0].strip(); desc = parts[1].strip() if len(parts) > 1 else ""
        collected_cmds.append((cmd, desc))
  if collected_cmds:
    seen_cmds = set(); ordered = []
    for cmd, desc in collected_cmds:
      if cmd not in seen_cmds:
        seen_cmds.add(cmd); ordered.append((cmd, desc))
    bullets = []
    for cmd, desc in ordered:
      bullets.append(f"- {cmd}" + (f": {desc}" if desc else ""))
    make_items.append({"instruction": "What are the make commands in section '3. Run the Complete Pipeline'?", "response": "\n".join(bullets)})
    for cmd, desc in ordered[:8]:
      q = f"What does '{cmd}' do?"
      answer = desc if desc else "Runs the corresponding step of the training/deployment pipeline defined in the Makefile."
      make_items.append({"instruction": q, "response": answer})

  try:
    with open(MAKEFILE_PATH, 'r', encoding='utf-8') as mf:
      mk_text = mf.read()
    target_lines = re.findall(r"^([a-zA-Z0-9_.-]+):\s*$", mk_text, flags=re.MULTILINE)
    qtype_match = re.search(r"^QTYPE\?=\s*([^\n]+)$", mk_text, flags=re.MULTILINE)
    merged_dir_match = re.search(r"^MERGED_DIR=([^\n]+)$", mk_text, flags=re.MULTILINE)
    vars_info = []
    if qtype_match:
      vars_info.append(f"QTYPE default is {qtype_match.group(1).strip()} (quantization for GGUF)")
    if merged_dir_match:
      vars_info.append(f"MERGED_DIR is {merged_dir_match.group(1).strip()} (path to merged safetensors model)")
    if target_lines:
      make_items.append({
        "instruction": "List available Makefile targets and important variables.",
        "response": "Targets: " + ", ".join(target_lines) + (". " + "; ".join(vars_info) if vars_info else "")
      })
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

  # Merge and dedupe
  all_items = []
  seen_instr = set()
  for d in items + meta_items + key_items + make_items:
    inst = d.get("instruction", "").strip(); resp = d.get("response", "").strip()
    if not inst or not resp:
      continue
    key = inst.lower()
    if key in seen_instr:
      continue
    seen_instr.add(key)
    all_items.append({"instruction": inst, "response": resp})
  return all_items


def _estimate_chars_per_token():
  # Simple heuristic: ~4 chars per token for English-ish text
  try:
    return float(os.environ.get("CHARS_PER_TOKEN", "4.0"))
  except Exception:
    return 4.0


def _chunk_long_text(text: str, max_chars: int, overlap: int = 40):
  text = (text or "").strip()
  if max_chars <= 0 or len(text) <= max_chars:
    return [text]
  chunks = []
  start = 0
  n = len(text)
  # Ensure positive progress even if overlap is large relative to max_chars
  effective_overlap = max(0, min(overlap, max_chars - 1))
  safety_counter = 0
  safety_limit = 100000  # hard cap to avoid infinite loops on pathological inputs
  while start < n and safety_counter < safety_limit:
    safety_counter += 1
    end = min(n, start + max_chars)
    cut = end
    # Try to cut on whitespace/punctuation within the last 80 chars, but not too close to start
    for i in range(end, max(start + 1, end - 80), -1):
      if text[i-1] in " .,;:!?\n\t-]})" and (i - start) > max(1, min(40, max_chars // 2)):
        cut = i
        break
    # Guarantee forward progress
    if cut <= start:
      cut = min(n, start + max(1, max_chars))
    chunk = text[start:cut].strip()
    chunks.append(chunk)
    if cut >= n:
      break
    # Compute next start with bounded overlap; ensure it advances
    next_start = max(0, cut - effective_overlap)
    if next_start <= start:
      next_start = min(n, start + max(1, max_chars // 2))
    start = next_start
  # Deduplicate possible overlaps at ends
  cleaned = []
  for c in chunks:
    if not cleaned or c != cleaned[-1]:
      cleaned.append(c)
  return cleaned


def _chunk_example(inst: str, resp: str, max_len_tokens: int):
  # Reserve a budget for prompt tokens; remaining for response
  # Prompt is: "### Instruction:\n{inst}\n\n### Response:\n" → approximate tokens
  chars_per_tok = _estimate_chars_per_token()
  prompt_overhead_tokens = 32  # rough
  # Instruction budget; if instruction is huge, split it too
  max_inst_tokens = max(16, int(max_len_tokens * 0.35))
  max_resp_tokens = max(32, max_len_tokens - prompt_overhead_tokens - max_inst_tokens)
  max_inst_chars = int(max_inst_tokens * chars_per_tok)
  max_resp_chars = int(max_resp_tokens * chars_per_tok)

  inst_chunks = _chunk_long_text(inst, max_inst_chars, overlap=20)
  result = []
  for i_inst, inst_chunk in enumerate(inst_chunks, start=1):
    inst_suffix = f" (part {i_inst}/{len(inst_chunks)})" if len(inst_chunks) > 1 else ""
    inst_final = f"{inst_chunk}{inst_suffix}".strip()
    resp_chunks = _chunk_long_text(resp, max_resp_chars, overlap=60)
    if len(resp_chunks) == 1:
      result.append({"instruction": inst_final, "response": resp_chunks[0]})
    else:
      for i_resp, resp_chunk in enumerate(resp_chunks, start=1):
        resp_head = f"[Response part {i_resp}/{len(resp_chunks)}]\n"
        result.append({"instruction": inst_final, "response": resp_head + resp_chunk})
  return result


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="dataset.txt", help="Path to output JSONL dataset file")
  ap.add_argument("--duplicates", type=int, default=10, help="Repeat examples N times for tiny datasets")
  ap.add_argument("--max-len", type=int, default=None, dest="max_len", help="MAX_LEN in tokens; defaults from env MAX_LEN or 128")
  args = ap.parse_args()

  # Resolve MAX_LEN to guide chunking
  max_len_tokens = args.max_len if args.max_len is not None else int(os.environ.get("MAX_LEN", "128"))

  try:
    with open(README_PATH, "r", encoding="utf-8") as f:
      readme_text = f.read()
  except Exception:
    readme_text = None

  items = parse_readme_for_train_data(readme_text) if readme_text else []
  if not items:
    print("[INFO] Falling back to default examples (README.md not found or no train_data block detected).")
    items = DEFAULT_EXAMPLES

  # Apply duplicates first (so chunking creates more varied ordering across duplicates)
  items = items * max(1, args.duplicates)

  # Chunk long instruction/response pairs according to max_len_tokens
  expanded = []
  for ex in items:
    expanded.extend(_chunk_example(ex["instruction"], ex["response"], max_len_tokens))

  # Write JSONL
  n = 0
  with open(args.out, "w", encoding="utf-8") as out:
    for ex in expanded:
      json.dump({"instruction": ex["instruction"], "response": ex["response"]}, out, ensure_ascii=False)
      out.write("\n")
      n += 1
  print(f"[OK] Wrote {n} examples to {args.out} (from {len(items)} base items; MAX_LEN={max_len_tokens})")


if __name__ == "__main__":
  main()
