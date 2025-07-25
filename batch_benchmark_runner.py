"""
Efficient batch composition benchmark runner with JSON caching and dynamic model loading.
"""

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from sglang_batch_latency import (
    ServerArgs, load_model, prepare_synthetic_inputs_for_latency_test,
    latency_test_run_once, PortArgs, _set_envs_and_config
)


class BatchBenchmarkRunner:
    """
    Efficient benchmark runner that caches models and results to avoid expensive re-computation.
    """
    
    def __init__(self, cache_dir: str = "./benchmark_cache"):
        self.cache_dir = cache_dir
        self.model_runner = None
        self.tokenizer = None
        self.current_model_path = None
        self.current_server_args = None
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, data: Any) -> str:
        """Generate a cache key for the given data."""
        return hashlib.md5(str(sorted(data.items()) if isinstance(data, dict) else str(data)).encode()).hexdigest()
    
    def _load_model_if_needed(self, model_path: str, server_args: Dict[str, Any]):
        """Load model only if it's different from the currently loaded one."""
        server_args_key = self._get_cache_key(server_args)
        
        if (self.model_runner is None or 
            self.current_model_path != model_path or 
            self.current_server_args != server_args_key):
            
            print(f"Loading model: {model_path}")
            
            # Set up server arguments
            server_args_obj = ServerArgs(
                model_path=model_path,
            )
            for key, value in server_args.items():
                if hasattr(server_args_obj, key):
                    setattr(server_args_obj, key, value)
            
            # Set environment and config
            _set_envs_and_config(server_args_obj)
            port_args = PortArgs.init_new(server_args_obj)
            
            # Load model
            self.model_runner, self.tokenizer = load_model(server_args_obj, port_args, 0)
            self.current_model_path = model_path
            self.current_server_args = server_args_key
            self.server_args_obj = server_args_obj
            
            print("Model loaded successfully")
    
    def _get_result_cache_path(self, cache_key: str) -> str:
        """Get the cache file path for results."""
        return os.path.join(self.cache_dir, f"results_{cache_key}.json")
    
    def _load_cached_results(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Load cached results if they exist."""
        cache_path = self._get_result_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                return data.get('results', data)
            except Exception as e:
                print(f"Failed to load cache: {e}")
        return None
    
    def _save_results_to_cache(self, cache_key: str, results: List[Dict[str, Any]], metadata: Dict[str, Any] = None):
        """Save results to cache with metadata."""
        cache_path = self._get_result_cache_path(cache_key)
        try:
            cache_data = {
                'metadata': metadata or {},
                'results': results,
                'cache_key': cache_key
            }
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def run_comprehensive_sweep(
        self,
        model_path: str,
        max_batch_size: int = 64,
        max_input_tokens: int = 16384,
        output_len: int = 32,
        server_args: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run comprehensive benchmark sweeping batch sizes and input lengths.
        
        Args:
            model_path: Path to the model
            max_batch_size: Maximum batch size to test (will test powers of 2 up to this)
            max_input_tokens: Maximum input length to test (will test powers of 2 up to this)
            output_len: Output sequence length
            server_args: Optional server arguments
            use_cache: Whether to use cached results
        
        Returns:
            List of benchmark results
        """
        if server_args is None:
            server_args = {}
        
        # Generate batch sizes as powers of 2: [1, 2, 4, 8, 16, 32, 64]
        batch_sizes = []
        power = 0
        while 2**power <= max_batch_size:
            batch_sizes.append(2**power)
            power += 1
        
        # Generate input lengths as powers of 2: [1, 2, 4, 8, ..., 16384]
        input_lens = []
        power = 0
        while 2**power <= max_input_tokens:
            input_lens.append(2**power)
            power += 1
        
        print(f"Batch sizes to test: {batch_sizes}")
        print(f"Input lengths to test: {input_lens}")
        
        # Generate cache key
        cache_data = {
            'model_path': model_path,
            'batch_sizes': batch_sizes,
            'input_lens': input_lens,
            'output_len': output_len,
            'server_args': server_args,
            'benchmark_type': 'comprehensive_sweep'
        }
        cache_key = self._get_cache_key(cache_data)
        
        # Try to load from cache
        if use_cache:
            cached_results = self._load_cached_results(cache_key)
            if cached_results is not None:
                print(f"Using cached results for comprehensive sweep")
                return cached_results
        
        # Load model if needed
        server_args['cuda_graph_max_bs'] = max_batch_size
        self._load_model_if_needed(model_path, server_args)
        
        # Run benchmarks with error handling
        results = []
        failed_configs = []
        
        total_configs = len(batch_sizes) * len(input_lens)
        current_config = 0
        
        for batch_size in batch_sizes:
            for input_len in input_lens:
                current_config += 1
                print(f"Running config {current_config}/{total_configs}: batch_size={batch_size}, input_len={input_len}, output_len={output_len}")
                
                try:
                    reqs = prepare_synthetic_inputs_for_latency_test(batch_size, input_len)
                    result = latency_test_run_once(
                        run_name=f"sweep_bs{batch_size}_il{input_len}",
                        model_runner=self.model_runner,
                        rank_print=lambda *args, **kwargs: None,
                        reqs=reqs,
                        batch_size=batch_size,
                        input_len=input_len,
                        output_len=output_len,
                        device=self.server_args_obj.device,
                        log_decode_step=0,
                        profile=False,
                        profile_filename_prefix=""
                    )
                    
                    if result is not None:
                        results.append(result)
                        decode_info = ""
                        if 'median_decode_latency' in result:
                            decode_info = f", decode={result['median_decode_latency']:.4f}s, decode_throughput={result['median_decode_throughput']:.1f} tok/s"
                        kv_cache_info = ""
                        if 'predicted_kv_usage_in_gb' in result:
                            kv_cache_info = f", kv_cache={result['predicted_kv_usage_in_gb']:.3f}GB"
                        print(f"  ✓ Success: prefill={result['prefill_latency']:.4f}s, prefill_throughput={result['prefill_throughput']:.1f} tok/s{decode_info}{kv_cache_info}")
                    else:
                        failed_configs.append({'batch_size': batch_size, 'input_len': input_len, 'reason': 'skipped_due_to_limits'})
                        print(f"  ⚠ Skipped due to resource limits")
                        
                except Exception as e:
                    failed_configs.append({'batch_size': batch_size, 'input_len': input_len, 'reason': str(e)})
                    print(f"  ✗ Failed: {str(e)}")
                    continue
        
        print(f"\nCompleted sweep: {len(results)} successful, {len(failed_configs)} failed")
        if failed_configs:
            print("Failed configurations:")
            for config in failed_configs:
                print(f"  bs={config['batch_size']}, il={config['input_len']}: {config['reason']}")
        
        # Save to cache
        if use_cache:
            metadata = {
                'benchmark_type': 'comprehensive_sweep',
                'model_path': model_path,
                'max_batch_size': max_batch_size,
                'max_input_tokens': max_input_tokens,
                'output_len': output_len,
                'batch_sizes_tested': batch_sizes,
                'input_lens_tested': input_lens,
                'successful_configs': len(results),
                'failed_configs': failed_configs
            }
            self._save_results_to_cache(cache_key, results, metadata)
        
        return results
    
    def run_variable_length_benchmark(
        self,
        model_path: str,
        input_ids_batches: List[List[List[int]]],
        output_lens: List[int],
        server_args: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        batch_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run benchmark with batches containing variable-length sequences.
        """
        if server_args is None:
            server_args = {}
        
        # Generate cache key (hash input_ids_batches structure, not content)
        batch_structures = []
        for batch in input_ids_batches:
            batch_structures.append([len(seq) for seq in batch])
        
        cache_data = {
            'model_path': model_path,
            'batch_structures': batch_structures,
            'output_lens': output_lens,
            'server_args': server_args,
            'benchmark_type': 'variable_length'
        }
        cache_key = self._get_cache_key(cache_data)
        
        # Try to load from cache
        if use_cache:
            cached_results = self._load_cached_results(cache_key)
            if cached_results is not None:
                print(f"Using cached results for variable length benchmark")
                return cached_results
        
        # Load model if needed
        max_batch_size = max(len(batch) for batch in input_ids_batches)
        server_args['cuda_graph_max_bs'] = max_batch_size
        self._load_model_if_needed(model_path, server_args)
        
        results = []
        failed_configs = []
        
        for batch_idx, input_ids_batch in enumerate(input_ids_batches):
            batch_name = batch_names[batch_idx] if batch_names else f"batch_{batch_idx}"
            batch_size = len(input_ids_batch)
            input_lengths = [len(seq) for seq in input_ids_batch]
            
            print(f"Running {batch_name}: batch_size={batch_size}, lengths={input_lengths}")
            
            for output_len in output_lens:
                try:
                    # Prepare requests with custom input_ids
                    reqs = prepare_synthetic_inputs_for_latency_test(
                        batch_size=batch_size,
                        input_len=0,  # Not used when input_ids_list is provided
                        input_ids_list=input_ids_batch
                    )
                    
                    # Run benchmark
                    result = latency_test_run_once(
                        run_name=f"var_len_{batch_name}",
                        model_runner=self.model_runner,
                        rank_print=lambda *args, **kwargs: None,
                        reqs=reqs,
                        batch_size=batch_size,
                        input_len=int(np.mean(input_lengths)),  # Average for reporting
                        output_len=output_len,
                        device=self.server_args_obj.device,
                        log_decode_step=0,
                        profile=False,
                        profile_filename_prefix=""
                    )
                    
                    if result is not None:
                        result['batch_name'] = batch_name
                        result['input_lengths'] = input_lengths
                        result['length_distribution'] = {
                            'min': min(input_lengths),
                            'max': max(input_lengths),
                            'mean': float(np.mean(input_lengths)),
                            'std': float(np.std(input_lengths))
                        }
                        results.append(result)
                        decode_info = ""
                        if 'median_decode_latency' in result:
                            decode_info = f", decode={result['median_decode_latency']:.4f}s, decode_throughput={result['median_decode_throughput']:.1f} tok/s"
                        kv_cache_info = ""
                        if 'predicted_kv_usage_in_gb' in result:
                            kv_cache_info = f", kv_cache={result['predicted_kv_usage_in_gb']:.3f}GB"
                        print(f"  ✓ Success: prefill={result['prefill_latency']:.4f}s, prefill_throughput={result['prefill_throughput']:.1f} tok/s{decode_info}{kv_cache_info}")
                    else:
                        failed_configs.append({
                            'batch_name': batch_name,
                            'batch_size': batch_size,
                            'input_lengths': input_lengths,
                            'output_len': output_len,
                            'reason': 'skipped_due_to_limits'
                        })
                        print(f"  ⚠ Skipped due to resource limits")
                        
                except Exception as e:
                    failed_configs.append({
                        'batch_name': batch_name,
                        'batch_size': batch_size,
                        'input_lengths': input_lengths,
                        'output_len': output_len,
                        'reason': str(e)
                    })
                    print(f"  ✗ Failed: {str(e)}")
                    continue
        
        # Save to cache
        if use_cache:
            metadata = {
                'benchmark_type': 'variable_length',
                'model_path': model_path,
                'batch_structures': batch_structures,
                'batch_names': batch_names,
                'output_lens': output_lens,
                'successful_configs': len(results),
                'failed_configs': failed_configs
            }
            self._save_results_to_cache(cache_key, results, metadata)
        
        return results
    
    def run_skewed_batch_benchmark(
        self,
        model_path: str,
        batch_size: int,
        base_length: int,
        skew_patterns: List[str],
        output_len: int,
        server_args: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run benchmark comparing different skew patterns in batch composition.
        """
        if server_args is None:
            server_args = {}
            
        # Generate input batches for different skew patterns
        input_ids_batches = []
        batch_names = []
        
        for pattern in skew_patterns:
            if pattern == 'even':
                # All sequences have the same length
                lengths = [base_length] * batch_size
            elif pattern == 'heavy_tail':
                # Most short, few very long
                lengths = [base_length // 4] * (batch_size - 2) + [base_length * 2, base_length * 3]
            elif pattern == 'bimodal':
                # Two distinct groups
                half = batch_size // 2
                lengths = [base_length // 2] * half + [base_length * 2] * (batch_size - half)
            elif pattern == 'linear':
                # Linearly increasing lengths
                lengths = [base_length + i * (base_length // batch_size) for i in range(batch_size)]
            else:
                raise ValueError(f"Unknown skew pattern: {pattern}")
            
            # Generate input_ids for this pattern
            batch = [np.random.randint(0, 10000, length).tolist() for length in lengths]
            input_ids_batches.append(batch)
            batch_names.append(pattern)
        
        return self.run_variable_length_benchmark(
            model_path=model_path,
            input_ids_batches=input_ids_batches,
            output_lens=[output_len],
            server_args=server_args,
            use_cache=use_cache,
            batch_names=batch_names
        )
    
    def clear_cache(self):
        """Clear all cached results."""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
        print("Cache cleared")
    
    def list_cached_results(self) -> List[Dict[str, Any]]:
        """List all cached benchmark results."""
        cached_files = []
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.startswith('results_') and filename.endswith('.json'):
                    filepath = os.path.join(self.cache_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        cached_files.append({
                            'filename': filename,
                            'metadata': data.get('metadata', {}),
                            'num_results': len(data.get('results', []))
                        })
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")
        return cached_files


def analyze_sweep_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze comprehensive sweep results and extract insights."""
    if not results:
        return {}
    
    # Group by batch size and input length
    by_batch_size = {}
    by_input_len = {}
    
    for result in results:
        bs = result['batch_size']
        il = result['input_len']
        
        if bs not in by_batch_size:
            by_batch_size[bs] = []
        by_batch_size[bs].append(result)
        
        if il not in by_input_len:
            by_input_len[il] = []
        by_input_len[il].append(result)
    
    # Calculate statistics
    decode_results = [r for r in results if 'median_decode_latency' in r]
    
    analysis = {
        'total_experiments': len(results),
        'batch_size_analysis': {},
        'input_length_analysis': {},
        'overall_stats': {
            'prefill_latency': {
                'min': min(r['prefill_latency'] for r in results),
                'max': max(r['prefill_latency'] for r in results),
                'mean': float(np.mean([r['prefill_latency'] for r in results]))
            },
            'decode_latency': {
                'min': min(r['median_decode_latency'] for r in decode_results) if decode_results else 0,
                'max': max(r['median_decode_latency'] for r in decode_results) if decode_results else 0,
                'mean': float(np.mean([r['median_decode_latency'] for r in decode_results])) if decode_results else 0
            }
        }
    }
    
    # Analyze by batch size
    for bs, bs_results in by_batch_size.items():
        prefill_latencies = [r['prefill_latency'] for r in bs_results]
        decode_bs_results = [r for r in bs_results if 'median_decode_latency' in r]
        
        analysis['batch_size_analysis'][bs] = {
            'num_experiments': len(bs_results),
            'avg_prefill_latency': float(np.mean(prefill_latencies)),
            'avg_decode_latency': float(np.mean([r['median_decode_latency'] for r in decode_bs_results])) if decode_bs_results else 0,
            'prefill_throughput_range': {
                'min': min(r['prefill_throughput'] for r in bs_results),
                'max': max(r['prefill_throughput'] for r in bs_results)
            },
            'decode_throughput_range': {
                'min': min(r['median_decode_throughput'] for r in decode_bs_results) if decode_bs_results else 0,
                'max': max(r['median_decode_throughput'] for r in decode_bs_results) if decode_bs_results else 0
            }
        }
    
    # Analyze by input length
    for il, il_results in by_input_len.items():
        prefill_latencies = [r['prefill_latency'] for r in il_results]
        decode_il_results = [r for r in il_results if 'median_decode_latency' in r]
        
        analysis['input_length_analysis'][il] = {
            'num_experiments': len(il_results),
            'avg_prefill_latency': float(np.mean(prefill_latencies)),
            'avg_decode_latency': float(np.mean([r['median_decode_latency'] for r in decode_il_results])) if decode_il_results else 0,
            'prefill_throughput_range': {
                'min': min(r['prefill_throughput'] for r in il_results),
                'max': max(r['prefill_throughput'] for r in il_results)
            },
            'decode_throughput_range': {
                'min': min(r['median_decode_throughput'] for r in decode_il_results) if decode_il_results else 0,
                'max': max(r['median_decode_throughput'] for r in decode_il_results) if decode_il_results else 0
            }
        }
    
    return analysis


# Example usage
if __name__ == "__main__":
    runner = BatchBenchmarkRunner()
    
    # Comprehensive sweep with error handling
    print("Running comprehensive sweep...")
    model = "Qwen/Qwen3-4B"
    results = runner.run_comprehensive_sweep(
        model_path=model,
        max_batch_size=64,
        max_input_tokens=2**14,
        output_len=32,
        server_args={"load_format": "dummy"}
    )
    
    print(f"\nCompleted {len(results)} successful experiments")
    analysis = analyze_sweep_results(results)
    model = model.replace("/", "_").replace(" ", "_")
    with open(f"benchmark_data_{model}.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    # # List cached results
    # print("\nCached results:")
    # for cache_info in runner.list_cached_results():
    #     print(f"  {cache_info['filename']}: {cache_info['num_results']} results")