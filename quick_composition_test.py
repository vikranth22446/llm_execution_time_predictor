#!/usr/bin/env python3
"""
Quick Batch Composition Test Script

Predefined test configurations for common batch composition scenarios.
Easily test different combinations like [14K, 1K], [8K, 2K, 4K], etc.

Usage:
python quick_composition_test.py --model-path meta-llama/Meta-Llama-3-8B-Instruct --preset mixed_workload
python quick_composition_test.py --model-path meta-llama/Meta-Llama-3-8B-Instruct --preset custom --request-lengths "1024,2048,512,4096"
"""

import argparse
import subprocess
import sys
from typing import Dict, List, Tuple


PRESETS = {
    "mixed_workload": {
        "description": "Mixed workload with varying request sizes",
        "request_lengths": "1024,2048,512,1024,4096,1024,512,2048",
        "compositions": [
            "16000",           # Single large batch
            "14000,2000",      # Your suggested composition
            "12000,4000",      # Alternative split
            "8000,4000,4000",  # Three-way split
            "6000,5000,5000",  # More balanced
            "4000,4000,4000,4000"  # Four equal batches
        ]
    },
    
    "small_requests": {
        "description": "Mostly small requests (1K or less)",
        "request_lengths": "512,256,1024,512,768,512,256,1024,512",
        "compositions": [
            "8000",
            "6000,2000", 
            "4000,2000,2000",
            "3000,3000,2000",
            "2000,2000,2000,2000"
        ]
    },
    
    "large_requests": {
        "description": "Mostly large requests (2K+)",
        "request_lengths": "2048,4096,3072,2048,4096,2560,3584,2048",
        "compositions": [
            "24000",
            "20000,4000",
            "16000,8000", 
            "12000,6000,6000",
            "8000,8000,8000"
        ]
    },
    
    "extreme_variance": {
        "description": "High variance in request sizes",
        "request_lengths": "512,8192,1024,16384,512,4096,1024,512,32768",
        "compositions": [
            "64000",
            "48000,16000",
            "32000,16000,16000",
            "24000,20000,20000",
            "16000,16000,16000,16000"
        ]
    }
}


def run_composition_test(model_path: str, preset_name: str, custom_requests: str = None, 
                        output_len: int = 32, extra_args: List[str] = None) -> int:
    """Run the batch composition test with specified parameters."""
    
    if preset_name == "custom":
        if not custom_requests:
            print("Error: --request-lengths required when using 'custom' preset")
            return 1
        request_lengths = custom_requests
        # Default compositions for custom requests
        compositions = ["16000", "14000,2000", "8000,4000,4000", "6000,5000,5000"]
    else:
        if preset_name not in PRESETS:
            print(f"Error: Unknown preset '{preset_name}'. Available presets: {list(PRESETS.keys())}")
            return 1
        
        preset = PRESETS[preset_name]
        request_lengths = preset["request_lengths"]
        compositions = preset["compositions"]
        print(f"Using preset '{preset_name}': {preset['description']}")
    
    # Build command
    cmd = [
        sys.executable, "batch_composition_tester.py",
        "--model-path", model_path,
        "--request-lengths", request_lengths,
        "--output-len", str(output_len),
        "--result-filename", f"{preset_name}_composition_results.jsonl"
    ]
    
    # Add compositions
    cmd.extend(["--compositions"] + compositions)
    
    # Add any extra arguments
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Testing {len(compositions)} batch compositions...")
    print(f"Request lengths: {request_lengths}")
    print()
    
    # Run the test
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Test failed with return code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print("Error: batch_composition_tester.py not found in current directory")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Quick batch composition testing with presets")
    parser.add_argument("--model-path", required=True, help="Path to the model")
    parser.add_argument(
        "--preset", 
        choices=list(PRESETS.keys()) + ["custom"], 
        default="mixed_workload",
        help="Preset configuration to use"
    )
    parser.add_argument(
        "--request-lengths",
        help="Custom request lengths (required if preset=custom)"
    )
    parser.add_argument("--output-len", type=int, default=32, help="Output length for all requests")
    parser.add_argument("--load-format", default="auto", help="Model loading format")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup")
    
    args, extra_args = parser.parse_known_args()
    
    # List available presets if requested
    if args.preset == "list":
        print("Available presets:")
        for name, config in PRESETS.items():
            print(f"  {name}: {config['description']}")
        return 0
    
    # Add common server args to extra_args if not already specified
    if "--load-format" not in extra_args and args.load_format:
        extra_args.extend(["--load-format", args.load_format])
    
    if args.no_warmup:
        extra_args.append("--no-warmup")
    
    return run_composition_test(
        args.model_path, 
        args.preset, 
        args.request_lengths,
        args.output_len,
        extra_args
    )


if __name__ == "__main__":
    sys.exit(main())