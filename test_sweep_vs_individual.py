#!/usr/bin/env python3
"""
Test script to compare results from batch sweep vs individual runs.
This script runs the same configuration using both methods and compares the results.
"""

import json
import tempfile
import os
from batch_benchmark_runner import BatchBenchmarkRunner
from sglang_batch_latency import main as sglang_main, ServerArgs, BenchArgs
import argparse

def run_individual_benchmark(model_path, batch_size, input_len, output_len, server_args_dict=None):
    """Run individual benchmark using sglang_batch_latency directly."""
    
    # Create temporary file for results  
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_result_file = f.name
    
    try:
        # Set up server args
        server_args = ServerArgs(model_path=model_path)
        if server_args_dict:
            for key, value in server_args_dict.items():
                if hasattr(server_args, key):
                    setattr(server_args, key, value)
        
        # Set up bench args
        bench_args = BenchArgs(
            run_name="individual_test",
            batch_size=(batch_size,),
            input_len=(input_len,),
            output_len=(output_len,),
            result_filename=temp_result_file
        )
        
        # Run the benchmark
        sglang_main(server_args, bench_args)
        
        # Read results
        results = []
        if os.path.exists(temp_result_file):
            with open(temp_result_file, 'r') as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line.strip()))
        
        return results[0] if results else None
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_result_file):
            os.unlink(temp_result_file)

def run_sweep_benchmark(model_path, batch_size, input_len, output_len, server_args_dict=None):
    """Run benchmark using the batch runner sweep functionality."""
    
    runner = BatchBenchmarkRunner()
    
    # Clear cache to ensure fresh run
    runner.clear_cache()
    
    # Run comprehensive sweep with single data point
    results = runner.run_comprehensive_sweep(
        model_path=model_path,
        max_batch_size=batch_size,
        max_input_tokens=input_len,
        output_len=output_len,
        server_args=server_args_dict or {},
        use_cache=False  # Don't use cache for this test
    )
    
    # Find the matching configuration
    for result in results:
        if (result['batch_size'] == batch_size and 
            result['input_len'] == input_len and
            result['output_len'] == output_len):
            return result
    
    return None

def compare_results(sweep_result, individual_result, tolerance=0.05):
    """Compare two benchmark results and check if they're similar within tolerance."""
    
    if not sweep_result or not individual_result:
        return False, "One or both results are None"
    
    # Key metrics to compare
    metrics_to_compare = [
        'prefill_latency',
        'prefill_throughput', 
        'median_decode_latency',
        'median_decode_throughput',
        'total_latency',
        'overall_throughput'
    ]
    
    differences = {}
    all_within_tolerance = True
    
    for metric in metrics_to_compare:
        if metric in sweep_result and metric in individual_result:
            sweep_val = sweep_result[metric]
            individual_val = individual_result[metric]
            
            # Calculate relative difference
            if sweep_val != 0:
                rel_diff = abs(sweep_val - individual_val) / sweep_val
            else:
                rel_diff = abs(individual_val)
            
            differences[metric] = {
                'sweep': sweep_val,
                'individual': individual_val,
                'rel_diff': rel_diff,
                'within_tolerance': rel_diff <= tolerance
            }
            
            if rel_diff > tolerance:
                all_within_tolerance = False
    
    return all_within_tolerance, differences

def main():
    # Test configuration
    model_path = "Qwen/Qwen3-4B"
    batch_size = 4
    input_len = 512
    output_len = 32
    server_args = {"load_format": "dummy"}
    
    print(f"Testing benchmark consistency for:")
    print(f"  Model: {model_path}")
    print(f"  Batch size: {batch_size}")
    print(f"  Input length: {input_len}")
    print(f"  Output length: {output_len}")
    print()
    
    print("Running sweep benchmark...")
    sweep_result = run_sweep_benchmark(model_path, batch_size, input_len, output_len, server_args)
    
    print("Running individual benchmark...")
    individual_result = run_individual_benchmark(model_path, batch_size, input_len, output_len, server_args)
    
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    if sweep_result:
        print(f"\nSweep result:")
        for key in ['prefill_latency', 'prefill_throughput', 'median_decode_latency', 'median_decode_throughput']:
            if key in sweep_result:
                print(f"  {key}: {sweep_result[key]}")
    else:
        print("Sweep result: None")
    
    if individual_result:
        print(f"\nIndividual result:")
        for key in ['prefill_latency', 'prefill_throughput', 'median_decode_latency', 'median_decode_throughput']:
            if key in individual_result:
                print(f"  {key}: {individual_result[key]}")
    else:
        print("Individual result: None")
    
    # Compare results
    is_consistent, comparison = compare_results(sweep_result, individual_result)
    
    print(f"\nConsistency check: {'PASS' if is_consistent else 'FAIL'}")
    
    if isinstance(comparison, dict):
        print("\nDetailed comparison:")
        for metric, data in comparison.items():
            status = "✓" if data['within_tolerance'] else "✗"
            print(f"  {status} {metric}:")
            print(f"      Sweep: {data['sweep']:.6f}")
            print(f"      Individual: {data['individual']:.6f}")
            print(f"      Rel. diff: {data['rel_diff']:.4f} ({data['rel_diff']*100:.2f}%)")
    else:
        print(f"Comparison failed: {comparison}")
    
    return is_consistent

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)