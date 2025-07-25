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
    for entry in tqdm(batch):
        gen = entry["labeled_generation"].split("</think>")[0]
        post = remove_step_tags(entry["labeled_generation"]).split("</think>")[1]

        step_tag = f"<step{entry['first_correct']+1}>"
        step_pos = gen.find(step_tag)

        gen1 = gen[:step_pos] if step_pos != -1 else gen
        gen2 = gen[step_pos:] if step_pos != -1 else ""

        gen1 = remove_step_tags(gen1)
        gen2_clean = remove_step_tags(gen2)

        step_splits = re.findall(r"(<step\d+>[^<]*)", gen2)
        step_tokens = [tokenizer.encode(remove_step_tags(s), add_special_tokens=False) for s in step_splits]
        step_decoded = [[tokenizer.decode([tid]) for tid in toks] for toks in step_tokens]

        results.append({
            **entry,
            "gen1": gen1,
            "gen2": gen2_clean,
            "gen1_tokens": [tokenizer.decode([tid]) for tid in tokenizer.encode(gen1)],
            "gen2_steps": step_splits,
            "gen2_step_tokens": step_decoded,
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
        question = item["problem"]
        generation = remove_step_tags(item["labeled_generation"])
        overthink_triggered = False
        mask_content = None
        epiphany_index = None

        step_token_buffer = token_list.copy()
        step_text_buffer = ""
        gen1_step_count = len(item["gen2_steps"])
        used_step_count = 0

        total_keyword_tokens = 0
        total_tokens = len(token_list)
        if total_tokens == 0:
            print(f"generation = {generation}")
            print(f"token_list = {token_list}")
            print(f"gen1 = {item['gen1_tokens']}")
            print(f"gen2 = {item['gen2_steps']}")

        for idx, (step_str, step_tokens) in enumerate(zip(item["gen2_steps"], item["gen2_step_tokens"])):
            step_text = "".join(step_tokens)
            step_text_buffer += step_text
            step_token_buffer += step_tokens
            used_step_count += 1

            step_token_len = len(step_tokens)
            total_tokens += step_token_len

            step_text_lower = step_text.lower()
            step_keyword_tokens = sum(step_text_lower.count(k) * l for k, l in keywords.items())
            total_keyword_tokens += step_keyword_tokens

            rer = len(item["gen2_steps"][:idx]) / (used_step_count + len(item["gen2_steps"][:idx]))
            omr = total_keyword_tokens / total_tokens

            score = calculate_overthink_score(rer, omr)

            if not overthink_triggered and score >= OVERTHINK_THRESHOLD:
                overthink_triggered = True
                epiphany_index = len(step_token_buffer)

            if overthink_triggered and score > OVERTHINK_THRESHOLD + 0.05:
                new_generation = "".join(step_token_buffer) + EPIPHANY + item["post_think"]
                new_generation = remove_extra_spacing(new_generation)

                segment_tokens = step_token_buffer[epiphany_index:]
                segment_text = "".join(segment_tokens)
                mask_content = segment_text

                sbt_d_dataset.append({
                    "input": question,
                    "output": new_generation,
                    "mask_content": mask_content
                })
                overthink_count += 1
                break

        else:
            if overthink_triggered:
                new_generation = "".join(step_token_buffer) + EPIPHANY + item["post_think"]
                new_generation = remove_extra_spacing(new_generation)

                segment_tokens = step_token_buffer[epiphany_index:]
                segment_text = "".join(segment_tokens)
                mask_content = segment_text

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
