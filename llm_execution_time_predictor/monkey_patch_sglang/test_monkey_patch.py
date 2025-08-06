#!/usr/bin/env python3

import os

# Set environment variables
os.environ["SGLANG_ENABLE_PROFILING"] = "1"
os.environ["SGLANG_PROFILE_OUTPUT"] = "model_runner_profile.json"

# Add current directory to PYTHONPATH for subprocesses
current_dir = os.path.dirname(os.path.abspath(__file__))
existing_pythonpath = os.environ.get("PYTHONPATH", "")
if existing_pythonpath:
    os.environ["PYTHONPATH"] = f"{current_dir}:{existing_pythonpath}"
else:
    os.environ["PYTHONPATH"] = current_dir

print(f"Main process {os.getpid()} starting")

import working_profiler

if __name__ == "__main__":
    # Clear existing profiling data
    import glob

    for old_file in glob.glob("model_runner_profile*.json"):
        os.remove(old_file)

    import sglang as sgl

    print("Creating SGLang Engine...")
    llm = sgl.Engine(model_path="qwen/qwen2.5-0.5b-instruct")
    import pandas as pd

    df = pd.read_csv("splitwise_code.csv")

    print("Running inference...")
    prompts = ["Hello, my name is", "The capital of France is", "The future of AI is"]
    outputs = llm.generate(
        prompts, {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 50}
    )
    llm.shutdown()
