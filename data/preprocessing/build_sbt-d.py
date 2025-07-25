import os
import re
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from concurrent.futures import ProcessPoolExecutor

print(f'Start building the SBT-D dataset...')
# ─── Config ─────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
TOKENIZER_PATH = "models/Qwen/Qwen2.5-Math-1.5B-Instruct" 
JSONL_PATH = "data/baseline/openr1_math_92k_augmented.jsonl"
KEYWORD_PATH = "prompts/keyword.txt" 
SBT_D_PATH = "data/SBT/SBT-D.jsonl" 
EPIPHANY = "Wait, my answer is too verbose, let me answer it more concisely."
OVERTHINK_THRESHOLD = float(input("Please choose the overthink OVERTHINK_THRESHOLD (e.g., 0.2): "))
print(f"[INFO] OVERTHINK_THRESHOLD set to: {OVERTHINK_THRESHOLD}")
# ───────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# ===========================
# UTILITIES
# ===========================
def load_keywords_and_token_lengths(path, tokenizer):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    keyword_dict = {}
    for line in lines:
        keyword = line.strip()
        if not keyword:
            continue
        norm_keyword = keyword.lower()
        if norm_keyword not in keyword_dict:
            token_count = len(tokenizer.encode(keyword, add_special_tokens=False))
            keyword_dict[norm_keyword] = token_count
    return keyword_dict

def remove_step_tags(text):
    return re.sub(r'<step\d+> ?', '', text)

def remove_extra_spacing(text):
    return re.sub(r'\n\n +', '\n\n', text)

def calculate_overthink_score(ratio_correct, ratio_keyword):
    return 0.9 * (1 - ratio_correct) + 0.1 * ratio_keyword

# ===========================
# DATA LOADING
# ===========================
def load_jsonl_dataset(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]
    print(f"[INFO] Loaded {len(data)} samples from {path}")
    return data

# ===========================
# TOKENIZATION
# ===========================
def tokenize_batch(batch, keywords):
    results = []
    for entry in batch:
        gen = entry["labeled_generation"].split("</think>")[0]
        post = remove_step_tags(entry["labeled_generation"]).split("</think>")[1]

        # Step boundary
        step_tag = f"<step{entry['first_correct']+1}>"
        step_pos = gen.find(step_tag)

        gen1 = gen[:step_pos] if step_pos != -1 else gen
        gen2 = gen[step_pos:] if step_pos != -1 else ""

        gen1 = remove_step_tags(gen1)
        gen2 = remove_step_tags(gen2)

        gen1_tokens = tokenizer.encode(gen1)
        gen2_tokens = tokenizer.encode(gen2)

        gen1_decoded = [tokenizer.decode([tid]) for tid in gen1_tokens]
        gen2_decoded = [tokenizer.decode([tid]) for tid in gen2_tokens]

        results.append({
            **entry,
            "gen1": gen1,
            "gen2": gen2,
            "gen1_tokens": gen1_decoded,
            "gen2_tokens": gen2_decoded,
            "post_think": post,
        })
    return results

def parallel_tokenize(data, keywords, batch_size=128):
    batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
    with ProcessPoolExecutor() as executor:
        results = executor.map(tokenize_batch, batches, [keywords]*len(batches))
    return [item for batch in results for item in batch]

# ===========================
# SBT-D LOGIC
# ===========================
def build_sbt_d_data(entries, keywords, OVERTHINK_THRESHOLD):
    sbt_d_dataset = []
    overthink_count = 0
    no_overthink_count = 0

    for item in tqdm(entries):
        token_list = item["gen1_tokens"].copy()
        total_len = len(token_list)
        question = item["question"]
        labeled_generation = item["generation"]
        generation = remove_step_tags(labeled_generation)
        overthink_triggered = False
        mask_content = None
        epiphany_index = None  # record length when epiphany first hits

        for tok in item["gen2_tokens"]:
            total_len += 1
            token_list.append(tok)

            merged_text = "".join(token_list).lower()
            rer = len(item["gen1_tokens"]) / total_len

            keyword_tokens = sum(
                merged_text.count(k) * l for k, l in keywords.items()
            )
            omr = keyword_tokens / total_len

            score = calculate_overthink_score(rer, omr)

            if not overthink_triggered and score >= OVERTHINK_THRESHOLD:
                overthink_triggered = True
                epiphany_index = len(token_list)

            if overthink_triggered and score > OVERTHINK_THRESHOLD + 0.05:
                new_generation = "".join(token_list) + EPIPHANY + item["post_think"]
                new_generation = remove_extra_spacing(new_generation)

                segment_tokens = token_list[epiphany_index:]
                segment_text = "".join(segment_tokens)
                mask_content = extract_mask_content(segment_text)

                sbt_d_dataset.append({
                    "input": question,
                    "output": new_generation,
                    "mask_content": mask_content
                })
                overthink_count += 1
                break

        else:
            if overthink_triggered:
                new_generation = "".join(token_list) + EPIPHANY + item["post_think"]
                new_generation = remove_extra_spacing(new_generation)

                segment_tokens = token_list[epiphany_index:]
                segment_text = "".join(segment_tokens)
                mask_content = extract_mask_content(segment_text)

                sbt_d_dataset.append({
                    "input": question,
                    "output": new_generation,
                    "mask_content": mask_content
                })
                overthink_count += 1
            else:
                sbt_d_dataset.append({
                    "input": question,
                    "output": generation
                })
                no_overthink_count += 1

    return sbt_d_dataset, overthink_count, no_overthink_count


# ===========================
# SAVE FUNCTION
# ===========================
def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# ===========================
# MAIN
# ===========================
def main():
    dataset = load_jsonl_dataset(JSONL_PATH)
    keywords = load_keywords_and_token_lengths(KEYWORD_PATH, tokenizer)
    tokenized_data = parallel_tokenize(dataset, keywords)
    sbt_d_data, overthink_count, no_overthink_count = build_sbt_d_data(tokenized_data, keywords, OVERTHINK_THRESHOLD)

    output_file = SBT_D_PATH
    save_jsonl(sbt_d_data, output_file)

    total = overthink_count + no_overthink_count
    print(f"[RESULT] Processed {len(sbt_d_data)} entries. Saved to {output_file}")
    print(f"[RESULT] Overthink Count: {overthink_count} [{overthink_count / total:.2%}]")
    print(f"[RESULT] No-Overthink Count: {no_overthink_count} [{no_overthink_count / total:.2%}]")


if __name__ == "__main__":
    main()
