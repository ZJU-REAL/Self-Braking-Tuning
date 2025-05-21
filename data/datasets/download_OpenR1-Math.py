'''
During the implementation of Self-Braking Tuning, we curate a dataset of 92K high-quality instances from OpenR1-Math [1] by applying a 16K context length limit and filtering out problematic samples (e.g., those containing multiple </think> tags).
In fact, we also manually filtered out a portion of the data, but the remaining dataset is already sufficient to demonstrate the effectiveness of Self-Braking Tuning.
'''
import os
import json
from datasets import load_dataset
from transformers import AutoTokenizer

# ─── Config ────────────────────────────────────────────────────────────────
MODEL_PATH      = "/home/zhaohaoran/stopLLM/model/Qwen/Qwen2.5-Math-7B-Instruct"
MAX_TOKENS      = 16384
OUTPUT_DIR      = "data/baseline"
OUTPUT_FILENAME = "openr1_math_92k.jsonl"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# 1. Load the ≈92K split
ds = load_dataset("open-r1/OpenR1-Math-220k", "default")["train"]

# 2. Extract only the fields you want + pull out the assistant message
def extract_fields(ex):
    gen = next((m["content"] for m in ex["messages"] if m["role"] == "assistant"), "")
    return {
        "problem":    ex["problem"],
        "solution":   ex["solution"],
        "answer":     ex["answer"],
        "uuid":       ex["uuid"],
        "generation": gen
    }

pruned = ds.map(
    extract_fields,
    remove_columns=ds.column_names
)

# 3. Filter to exactly one </think>
filtered = pruned.filter(lambda ex: ex["generation"].count("</think>") == 1)

# 4. Load tokenizer for length check
def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer

tokenizer = load_tokenizer(MODEL_PATH)

# 5. Define your full-message template & token‑length filter
def within_token_limit(ex):
    full = (
        "<|im_start|>user\n"      + ex["problem"]    +
        "<|im_end|>\n<|im_start|>assistant\n" + ex["generation"] +
        "<|im_end|>"
    )
    tokens = tokenizer(full, return_tensors="pt", truncation=False).input_ids
    return tokens.shape[1] < MAX_TOKENS

# 6. Apply token‑length filtering
final_ds = filtered.filter(within_token_limit)

# 7. Export to JSONL
def add_step_tags(text):
    parts = text.split('\n\n')
    tagged = [f"<step{i+1}>{part}" for i, part in enumerate(parts)]
    return ''.join(tagged)

with open(out_path, "w", encoding="utf-8") as fout:
    for rec in final_ds:
        modified_rec = rec.copy()
        modified_rec["generation"] = add_step_tags(modified_rec["generation"])
        fout.write(json.dumps(modified_rec, ensure_ascii=False) + "\n")

print(f"▶️ Exported {len(final_ds)} examples to {out_path}")
