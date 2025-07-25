import json
import re
from transformers import AutoTokenizer
from tqdm import tqdm

print(f'Start building the SBT-E dataset...')

# ===========================
# CONFIGURATION
# ===========================
JSONL_PATH = "data/baseline/openr1_math_92k_augmented.jsonl"         # Path to the augmented dataset
KEYWORD_PATH = "prompts/keyword.txt"                                 # Path to keyword list
SBT_E_PATH = "data/SBT/SBT-D.jsonl"                                      # Output path for the SBT-E formatted data
TOKENIZER_PATH = "models/Qwen/Qwen2.5-Math-1.5B-Instruct"            # Tokenizer source (local or remote)
OVERTHINK_THRESHOLD = float(input("Please choose the overthink threshold (e.g., 0.2): "))  # Threshold for overthinking detection
EPIPHANY = "Wait, I've gotten the same answer multiple times, time to end the thinking."   # Text to insert at epiphany point


# ===========================
# UTILITIES
# ===========================
def remove_step_tags(text):
    """
    Remove all <stepX> tags from the text.
    These tags mark thinking steps but are not needed for token analysis.
    """
    return re.sub(r"<step\d+> ?", "", text, flags=re.DOTALL)


def load_keywords_and_token_lengths(path, tokenizer):
    """
    Load keywords from a file and calculate their token lengths using the tokenizer.

    Args:
        path (str): Path to keyword file.
        tokenizer: HuggingFace tokenizer instance.

    Returns:
        dict: Mapping from normalized keyword to token count.
    """
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


def compute_metrics(entry, tokenizer, keyword_token_map):
    """
    Compute rer, omr, and overthink score for each entry.
    Annotate with overthink tag based on score threshold.

    Args:
        entry (dict): A single dataset entry.
        tokenizer: Tokenizer to tokenize the generation.
        keyword_token_map (dict): Keyword-to-token-length mapping.

    Returns:
        dict: Updated entry with scores and tags.
    """
    generation = entry["generation"]
    total_steps = entry["total_steps"]
    first_solution_end = entry.get("first_solution_end", None)

    rer = float(first_solution_end) / total_steps if first_solution_end is not None else None
    entry["rer"] = rer

    plain_text = remove_step_tags(generation)
    tokens = tokenizer.encode(plain_text, add_special_tokens=False)
    total_tokens = len(tokens)

    generation_lower = plain_text.lower()
    keyword_token_total = 0
    for keyword, token_count in keyword_token_map.items():
        occurrences = generation_lower.count(keyword)
        keyword_token_total += occurrences * token_count

    omr = keyword_token_total / total_tokens if total_tokens > 0 else 0.0
    entry["omr"] = omr

    if rer is not None:
        overthink_score = 0.1 * omr + 0.9 * (1 - rer)
    else:
        overthink_score = None
    entry["overthink_score"] = overthink_score

    entry["overthink_tag"] = "Overthink" if overthink_score is not None and overthink_score > OVERTHINK_THRESHOLD else "No-Overthink"

    return entry


def extract_mask_content(after_step_text):
    """
    Extract the first two sentences after the epiphany step.
    
    Args:
        after_step_text (str): Text after <stepN> where overthinking ends.

    Returns:
        str: Masked summary content.
    """
    sentences = re.split(r'(?<=[.?!])\s+', after_step_text.strip())
    return ' '.join(sentences[:2])

# ===========================
# SBT-E LOGIC
# ===========================
def build_sbt_e_entry(entry):
    """
    Convert an entry into SBT-E format depending on overthinking status.

    Args:
        entry (dict): Annotated entry with overthink tag.

    Returns:
        dict or None: SBT-E entry with input/output (+ mask_content if applicable).
    """
    input_text = entry["question"]
    generation = entry["generation"]
    tag = entry.get("overthink_tag")
    second_solution_end = entry.get("second_solution_end")

    if tag == "No-Overthink":
        output_text = remove_step_tags(generation).strip()
        return {"input": input_text, "output": output_text}

    elif tag == "Overthink" and second_solution_end is not None:
        step_pattern = f"<step{second_solution_end + 1}>"
        if step_pattern not in generation:
            return {"input": input_text, "output": output_text}

        before, after = generation.split(step_pattern, 1)
        effective_think = remove_step_tags(before).strip()
        mask_content = extract_mask_content(remove_step_tags(after))
        conclusion_match = re.search(r"</think>(.*)", generation, re.DOTALL)
        conclusion = conclusion_match.group(1).strip() if conclusion_match else ""

        output_text = f"{effective_think} {EPIPHANY} {conclusion}".strip()
        return {
            "input": input_text,
            "output": output_text,
            "mask_content": mask_content
        }

    return None


# ===========================
# MAIN FUNCTION
# ===========================
def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    keyword_token_map = load_keywords_and_token_lengths(KEYWORD_PATH, tokenizer)

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        entries = [json.loads(line.strip()) for line in f]

    augmented_entries = []
    sbt_e_entries = []

    for entry in tqdm(entries, desc="Processing"):
        updated = compute_metrics(entry, tokenizer, keyword_token_map)
        augmented_entries.append(updated)

        sbt = build_sbt_e_entry(updated)
        if sbt:
            sbt_e_entries.append(sbt)

    total_samples = len(augmented_entries)
    overthink_count = sum(1 for e in augmented_entries if e.get("overthink_tag") == "Overthink")
    no_overthink_count = sum(1 for e in augmented_entries if e.get("overthink_tag") == "No-Overthink")

    print(f"Overthink Count: {overthink_count} [{overthink_count/total_samples:.2%}]")
    print(f"No-Overthink Count: {no_overthink_count} [{no_overthink_count/total_samples:.2%}]")

    # Save SBT-E formatted output
    with open(SBT_E_PATH, "w", encoding="utf-8") as f:
        for item in sbt_e_entries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done! Output written to → {SBT_E_PATH}")


# ===========================
# ENTRY POINT
# ===========================
if __name__ == "__main__":
    main()
