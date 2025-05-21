import os
import torch
from modelscope import snapshot_download

# 设置模型存储路径
model_path = './models'
model_dir = snapshot_download('Qwen/Qwen2.5-Math-1.5B-Instruct', cache_dir=model_path, revision='master')
# model_dir = snapshot_download('Qwen/Qwen2.5-Math-7B-Instruct', cache_dir=model_path, revision='master')
# model_dir = snapshot_download('LLM-Research/Meta-Llama-3.1-8B-Instruct', cache_dir=model_path, revision='master')
# model_dir = snapshot_download('LLM-Research/Llama-3.2-1B-Instruct', cache_dir=model_path, revision='master')


print("All the models have been downloaded!")
