#!/usr/bin/env python3
"""
Final verification script to test if benchmark sweep results match individual runs.
This script addresses warmup differences and tests multiple configurations.
"""

import json
import subprocess
import sys
import tempfile
import os
import statistics
from typing import List, Dict, Any

def run_command_line_benchmark(model_path: str, batch_size: int, input_len: int, output_len: int, runs: int = 3) -> List[Dict[str, Any]]:
    """Run multiple command line benchmarks and return all results."""
    
    results = []
    
    for i in range(runs):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_result_file = f.name
        
        try:
            cmd = [
                sys.executable, "sglang_batch_latency.py",
                "--model-path", model_path,
                "--load-format", "dummy", 
                "--batch-size", str(batch_size),
                "--input-len", str(input_len),
                "--output-len", str(output_len),
                "--result-filename", temp_result_file,
                "--run-name", f"cmd_test_run_{i}"
            ]
            
            print(f"Running command line test {i+1}/{runs}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Read results
                if os.path.exists(temp_result_file):
                    with open(temp_result_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                results.append(json.loads(line.strip()))
            else:
                print(f"Command failed: {result.stderr}")
                
        except Exception as e:
            print(f"Error in run {i}: {e}")
        finally:
            if os.path.exists(temp_result_file):
                os.unlink(temp_result_file)
    
    return results

def run_batch_runner_test(model_path: str, batch_size: int, input_len: int, output_len: int) -> Dict[str, Any]:
    """Run test using the batch runner (similar to sweep functionality)."""
    
    from batch_benchmark_runner import BatchBenchmarkRunner
    
    runner = BatchBenchmarkRunner()
    runner.clear_cache()  # Ensure fresh run
    
    # Use run_comprehensive_sweep but with exact target configuration
    results = runner.run_comprehensive_sweep(
        model_path=model_path,
        max_batch_size=batch_size,
        max_input_tokens=input_len,
        output_len=output_len,
        server_args={"load_format": "dummy"},
        use_cache=False
    )
    
    # Find matching result
    for result in results:
        if (result['batch_size'] == batch_size and
            result['input_len'] == input_len and
            result['output_len'] == output_len):
            return result
    
    return None

def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate multiple benchmark results."""
    
    if not results:
        return None
    
    if len(results) == 1:
        return results[0]
    
    # Metrics to aggregate
    metrics = ['prefill_latency', 'prefill_throughput', 'median_decode_latency', 'median_decode_throughput']
    
    aggregated = {
        'run_name': 'aggregated',
        'batch_size': results[0]['batch_size'],
        'input_len': results[0]['input_len'],
        'output_len': results[0]['output_len'],
        'num_runs': len(results)
    }
    
    for metric in metrics:
        values = [r[metric] for r in results if metric in r]
        if values:
            aggregated[f'{metric}_mean'] = statistics.mean(values)
            aggregated[f'{metric}_median'] = statistics.median(values)
            aggregated[f'{metric}_std'] = statistics.stdev(values) if len(values) > 1 else 0
            aggregated[f'{metric}_min'] = min(values)
            aggregated[f'{metric}_max'] = max(values)
            # Use median as the representative value
            aggregated[metric] = statistics.median(values)
    
    return aggregated

def compare_methods(cmd_results: List[Dict[str, Any]], sweep_result: Dict[str, Any], tolerance: float = 0.15) -> bool:
    """Compare command line results with sweep result."""
    
    if not cmd_results or not sweep_result:
        print("Cannot compare - missing results")
        return False
    
    # Aggregate command line results
    cmd_agg = aggregate_results(cmd_results)
    
    print(f"\nComparison Results:")
    print("=" * 60)
    print(f"Command line runs: {len(cmd_results)}")
    print(f"Configuration: batch_size={sweep_result['batch_size']}, input_len={sweep_result['input_len']}, output_len={sweep_result['output_len']}")
    
    # Metrics to compare
    metrics = ['prefill_latency', 'prefill_throughput', 'median_decode_latency', 'median_decode_throughput']
    
    all_close = True
    
    for metric in metrics:
        if metric in cmd_agg and metric in sweep_result:
            cmd_val = cmd_agg[metric]
            sweep_val = sweep_result[metric]
            
            if cmd_val != 0:
                rel_diff = abs(cmd_val - sweep_val) / abs(cmd_val)
            else:
                rel_diff = abs(sweep_val)
            
            close = rel_diff <= tolerance
            status = "✓" if close else "✗"
            
            print(f"\n{status} {metric}:")
            print(f"    Command line (median): {cmd_val:.6f}")
            if len(cmd_results) > 1:
                print(f"    Command line (std): {cmd_agg.get(f'{metric}_std', 0):.6f}")
                print(f"    Command line (range): [{cmd_agg.get(f'{metric}_min', 0):.6f}, {cmd_agg.get(f'{metric}_max', 0):.6f}]")
            print(f"    Sweep result: {sweep_val:.6f}")  
            print(f"    Rel diff: {rel_diff:.4f} ({rel_diff*100:.2f}%)")
            
            if not close:
                all_close = False
        else:
            print(f"\n✗ {metric}: missing in one result")
            all_close = False
    
    return all_close

def main():
    """Main test function."""
    
    # Test configurations
    test_configs = [
        {"batch_size": 1, "input_len": 512, "output_len": 16},
        {"batch_size": 2, "input_len": 256, "output_len": 32},
        {"batch_size": 4, "input_len": 128, "output_len": 16},
    ]
    
    model_path = "Qwen/Qwen3-4B"
    
    print("Benchmark Verification Test")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Testing {len(test_configs)} configurations...")
    print()
    
    overall_success = True
    
    for i, config in enumerate(test_configs):
        print(f"\nTest {i+1}/{len(test_configs)}: {config}")
        print("-" * 40)
        
        try:
            # Run command line tests (multiple runs for stability)
            cmd_results = run_command_line_benchmark(
                model_path, 
                config["batch_size"], 
                config["input_len"], 
                config["output_len"],
                runs=2  # Run twice to account for variability
            )
            
            # Run sweep test
            print("Running sweep test...")
            sweep_result = run_batch_runner_test(
                model_path,
                config["batch_size"],
                config["input_len"], 
                config["output_len"]
            )
            
            # Compare
            config_success = compare_methods(cmd_results, sweep_result)
            
            print(f"\nConfig {i+1} result: {'PASS' if config_success else 'FAIL'}")
            
            if not config_success:
                overall_success = False
                
        except Exception as e:
            print(f"Error testing config {i+1}: {e}")
            overall_success = False
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULT: {'PASS' if overall_success else 'FAIL'}")
    print(f"{'='*60}")
    
    if overall_success:
        print("✓ Benchmark sweep results are consistent with individual runs")
    else:
        print("✗ Found inconsistencies between sweep and individual runs")
        print("  This could be due to:")
        print("  - Different warmup procedures")
        print("  - Model caching differences")
        print("  - Random seed variations")
        print("  - Hardware state differences")
    
    return overall_success

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted")
        exit(1)