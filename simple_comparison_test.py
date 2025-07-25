#!/usr/bin/env python3
"""
Simplified test to compare sweep vs individual results using command line interface.
"""

import json
import subprocess
import tempfile
import os
import sys

def run_individual_command(model_path, batch_size, input_len, output_len):
    """Run individual benchmark using command line."""
    
    # Create temporary file for results
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_result_file = f.name
    
    try:
        # Build command
        cmd = [
            sys.executable, "sglang_batch_latency.py",
            "--model-path", model_path,
            "--load-format", "dummy",
            "--batch-size", str(batch_size),
            "--input-len", str(input_len), 
            "--output-len", str(output_len),
            "--result-filename", temp_result_file,
            "--run-name", "individual_test"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return None
        
        # Read results
        results = []
        if os.path.exists(temp_result_file):
            with open(temp_result_file, 'r') as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line.strip()))
        
        return results[0] if results else None
        
    except subprocess.TimeoutExpired:
        print("Command timed out")
        return None
    except Exception as e:
        print(f"Error running command: {e}")
        return None
    finally:
        # Clean up temp file
        if os.path.exists(temp_result_file):
            os.unlink(temp_result_file)

def run_simple_programmatic_test(model_path, batch_size, input_len, output_len):
    """Run a simple programmatic test using the latency_test_run_once function directly."""
    
    from sglang_batch_latency import (
        ServerArgs, load_model, prepare_synthetic_inputs_for_latency_test,
        latency_test_run_once, PortArgs, _set_envs_and_config
    )
    
    try:
        # Set up server arguments
        server_args = ServerArgs(model_path=model_path, load_format="dummy")
        server_args.cuda_graph_max_bs = batch_size
        
        # Set environment and config
        _set_envs_and_config(server_args)
        port_args = PortArgs.init_new(server_args)
        
        # Load model
        model_runner, tokenizer = load_model(server_args, port_args, 0)
        
        # Prepare inputs
        reqs = prepare_synthetic_inputs_for_latency_test(batch_size, input_len)
        
        # Run benchmark
        result = latency_test_run_once(
            run_name="programmatic_test",
            model_runner=model_runner,
            rank_print=lambda *args, **kwargs: None,  # Silent
            reqs=reqs,
            batch_size=batch_size,
            input_len=input_len,
            output_len=output_len,
            device=server_args.device,
            log_decode_step=0,
            profile=False,
            profile_filename_prefix=""
        )
        
        return result
        
    except Exception as e:
        print(f"Programmatic test failed: {e}")
        return None

def compare_results(result1, result2, label1="Result 1", label2="Result 2", tolerance=0.1):
    """Compare two results."""
    
    if not result1 or not result2:
        print(f"Cannot compare - one or both results are None")
        print(f"{label1}: {result1 is not None}")  
        print(f"{label2}: {result2 is not None}")
        return False
    
    # Key metrics to compare
    metrics = ['prefill_latency', 'prefill_throughput', 'median_decode_latency', 'median_decode_throughput']
    
    print(f"\n{label1} vs {label2} Comparison:")
    print("-" * 50)
    
    all_close = True
    for metric in metrics:
        if metric in result1 and metric in result2:
            val1 = result1[metric]
            val2 = result2[metric]
            
            if val1 != 0:
                rel_diff = abs(val1 - val2) / abs(val1)
            else:
                rel_diff = abs(val2)
            
            close = rel_diff <= tolerance
            status = "✓" if close else "✗"
            
            print(f"{status} {metric}:")
            print(f"    {label1}: {val1:.6f}")
            print(f"    {label2}: {val2:.6f}")
            print(f"    Rel diff: {rel_diff:.4f} ({rel_diff*100:.2f}%)")
            
            if not close:
                all_close = False
        else:
            print(f"✗ {metric}: missing in one of the results")
            all_close = False
    
    return all_close

def main():
    # Test configuration
    model_path = "Qwen/Qwen3-4B"
    batch_size = 2
    input_len = 256
    output_len = 16
    
    print(f"Testing configuration:")
    print(f"  Model: {model_path}")
    print(f"  Batch size: {batch_size}")
    print(f"  Input length: {input_len}")
    print(f"  Output length: {output_len}")
    print()
    
    # Test 1: Command line vs programmatic
    print("=" * 60)
    print("TEST: Command Line vs Programmatic")
    print("=" * 60)
    
    print("Running command line benchmark...")
    cmd_result = run_individual_command(model_path, batch_size, input_len, output_len)
    
    print("Running programmatic benchmark...")
    prog_result = run_simple_programmatic_test(model_path, batch_size, input_len, output_len)
    
    # Compare results
    success = compare_results(cmd_result, prog_result, "Command line", "Programmatic")
    
    print(f"\nOverall result: {'PASS' if success else 'FAIL'}")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted")
        exit(1)