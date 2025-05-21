"""
This script provides a simple and unified way to download several math benchmark datasets 
from the HuggingFace Hub. It is intended to help users easily reproduce our experiments 
by ensuring consistent and accessible dataset preparation.

The included datasets cover a variety of math tasks such as AIME24, AIME25, AMC23, GSM8K, 
and MATH500. Each dataset will be saved locally under the 'benchmarks/' directory.

Please ensure you have access to the datasets on HuggingFace and that your environment 
has internet connectivity.
"""
import os
from datasets import load_dataset

# Create a directory for benchmark datasets
BENCHMARK_DIR = "benchmarks"
os.makedirs(BENCHMARK_DIR, exist_ok=True)

def download_dataset(dataset_name, save_path, **kwargs):
    """
    Generic dataset download function using HuggingFace Datasets.

    Args:
        dataset_name (str): The HuggingFace dataset identifier.
        save_path (str): Local path to save the dataset.
        **kwargs: Additional keyword arguments passed to `load_dataset`.

    Raises:
        Exception: If dataset download fails.
    """
    try:
        print(f"Downloading {dataset_name}...")
        dataset = load_dataset(dataset_name, **kwargs)
        dataset.save_to_disk(save_path)
        print(f"{dataset_name} saved to {save_path}")
    except Exception as e:
        print(f"Failed to download {dataset_name}: {str(e)}")
        print("Please check your internet connection or verify the dataset name.")

def main():
    # AIME24 (Apache License 2.0)
    download_dataset(
        dataset_name="HuggingFaceH4/aime_2024",
        save_path=os.path.join(BENCHMARK_DIR, "aime24")
    )
    
    # AIME25 (Apache License 2.0)
    download_dataset(
        dataset_name="yentinglin/aime_2025",
        save_path=os.path.join(BENCHMARK_DIR, "aime25")
    )
    
    # AMC23 (Apache License 2.0)
    download_dataset(
        dataset_name="math-ai/amc23",
        save_path=os.path.join(BENCHMARK_DIR, "amc23")
    )
    
    # GSM8K (MIT License)
    download_dataset(
        dataset_name="gsm8k",
        name="main",
        save_path=os.path.join(BENCHMARK_DIR, "gsm8k")
    )
    
    # MATH500 (Apache License 2.0)
    download_dataset(
        dataset_name="HuggingFaceH4/MATH-500",
        save_path=os.path.join(BENCHMARK_DIR, "math500")
    )
if __name__ == "__main__":
    main()
