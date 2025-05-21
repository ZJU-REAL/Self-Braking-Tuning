import os
import json
import re
import numpy as np
from collections import Counter
import openai
from tqdm import tqdm

# ─── Configuration ─────────────────────────────────────────────
NUM_SAMPLES = 4  # Number of times to sample responses from the language model
MODEL = "gpt-4"  # Model to use for inference
PROMPT_PATH = "prompts/solution_identification.txt"  # Path to the prompt template
INPUT_JSONL = "data/baseline/openr1_math_92k.jsonl"  # Input JSONL file containing data
OUTPUT_JSONL = "data/baseline/openr1_math_92k_augmented.jsonl"  # Output file with added fields
# ───────────────────────────────────────────────────────────────

def analyze_reasoning(truncated_solution, solution, answer, model=MODEL):
    """
    Calls the language model API to analyze reasoning based on truncated solution.
    Returns multiple samples of model outputs.
    """
    api_key = os.getenv("APIKEY")
    if not api_key:
        raise ValueError("Please set the APIKEY environment variable")
    openai.api_key = api_key

    # Load prompt template
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # Format the prompt
    full_prompt = prompt_template.format(
        truncated_solution=truncated_solution,
        solution=solution,
        answer=answer
    )

    responses = []
    for i in range(NUM_SAMPLES):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.7
            )
            message = response['choices'][0]['message']['content']
            responses.append(message.strip())
        except Exception as e:
            responses.append(f"[Error during sampling {i+1}]: {e}")

    return responses


def extract_all_and_majority_vote(result_list, total_step):
    """
    Extracts all matched step numbers from the model outputs and determines the majority vote.
    If a tie occurs, returns the smallest step. Only returns steps within total_step range.
    """
    all_matches = []

    for result in result_list:
        matches = []
        # Regular expressions to extract step numbers in various formats
        patterns = [
            r'\*\*first_occurrence_step\*\*\s*[:：=]?\s*<?(?:step)?(\d+)>?',
            r'\*\*first_occurrence_step\*\*\s*[:：=]?\s*\*{0,2}(\d+)\*{0,2}',
            r'first_occurrence_step\s*[:：=]?\s*(?:<step)?(\d+)(?:>)?',
            r'first_occurrence_step\s*[:：=]?\s*<(?:step)?(\d+)>'
        ]
        for pat in patterns:
            match = re.findall(pat, result)
            if match:
                matches.extend([int(m) for m in match])
        all_matches.extend(matches)

    if not all_matches:
        return np.nan

    # Count occurrences and determine the majority
    counter = Counter(all_matches)
    most_common = counter.most_common()
    max_count = most_common[0][1]
    candidates = [num for num, count in most_common if count == max_count]
    final_candidate = min(candidates)

    return final_candidate if final_candidate <= int(total_step) else np.nan


def get_truncated_from_step(generation, from_step):
    """
    Returns the truncated generation starting from a specific step.
    """
    pattern = re.compile(rf"<step{from_step}>(.*?)((?=<step\d+>)|$)", re.DOTALL)
    match = pattern.search(generation)
    if not match:
        return ""
    
    truncated = generation.split(f"<step{from_step}>", 1)[-1]
    return f"<step{from_step}>" + truncated


def count_total_steps(generation):
    """
    Counts the number of <stepX> tags in the generation text,
    only up to the first occurrence of </think>.
    """
    # Truncate at the first </think> if present
    think_end_idx = generation.find("</think>")
    if think_end_idx != -1:
        truncated_text = generation[:think_end_idx]
    else:
        truncated_text = generation

    # Count <stepX> tags in the truncated part
    return len(re.findall(r"<step\d+>", truncated_text))



def process_jsonl(input_path, output_path):
    """
    Main processing function. Loads a JSONL file, analyzes the reasoning chain
    in two stages (first and second solution segments), and writes results with
    new fields `first_solution_end` and `second_solution_end`.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line.strip()) for line in f]

    output_lines = []
    for entry in tqdm(lines, desc="Processing"):
        generation = entry["generation"]
        solution = entry["solution"]
        answer = entry["answer"]

        total_steps = count_total_steps(generation)
        entry["total_steps"] = total_steps
        # First round of analysis
        responses_1 = analyze_reasoning(generation, solution, answer)
        first_solution_end = extract_all_and_majority_vote(responses_1, total_steps)
        entry["first_solution_end"] = int(first_solution_end) if not np.isnan(first_solution_end) else None

        # Truncate the generation from the next step onward
        if entry["first_solution_end"] is not None:
            new_truncated = get_truncated_from_step(generation, entry["first_solution_end"] + 1)
        else:
            new_truncated = ""

        # Second round of analysis
        if new_truncated.strip():
            responses_2 = analyze_reasoning(new_truncated, solution, answer)
            second_solution_end = extract_all_and_majority_vote(responses_2, total_steps)
            entry["second_solution_end"] = int(second_solution_end) if not np.isnan(second_solution_end) else None
        else:
            entry["second_solution_end"] = None

        output_lines.append(entry)

    # Save updated entries to output JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for item in output_lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Processing complete. {len(output_lines)} entries written to: {output_path}")


if __name__ == "__main__":
    process_jsonl(INPUT_JSONL, OUTPUT_JSONL)
