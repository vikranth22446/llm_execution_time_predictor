"""
Efficient batch composition benchmark runner with JSON caching and dynamic model loading.
"""

import hashlib
import json
import os
import multiprocessing

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from sglang_batch_latency import (
    load_model, prepare_synthetic_inputs_for_latency_test,
    latency_test_run_once, _set_envs_and_config
)
from sglang.srt.server_args import ServerArgs, PortArgs
from sglang.srt.distributed.parallel_state import destroy_distributed_environment
from sglang.srt.utils import kill_process_tree


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
    
    def _load_model_if_needed(self, model_path: str, server_args: Dict[str, Any], tp_rank: int = 0):
        """Load model only if it's different from the currently loaded one."""
        server_args_key = self._get_cache_key(server_args)
        
        if (self.model_runner is None or 
            self.current_model_path != model_path or 
            self.current_server_args != server_args_key):
            
            rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None
            rank_print(f"Loading model: {model_path}")
            
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
            self.model_runner, self.tokenizer = load_model(server_args_obj, port_args, tp_rank)
            self.current_model_path = model_path
            self.current_server_args = server_args_key
            self.server_args_obj = server_args_obj
            
            rank_print("Model loaded successfully")
    
    def _get_result_cache_path(self, cache_key: str) -> str:
        """Get the cache file path for results."""
        return os.path.join(self.cache_dir, f"results_{cache_key}.json")
    
    def _load_cached_data(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Load cached results if they exist."""
        cache_path = self._get_result_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                return data
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
    
    def _average_results(self, results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Average multiple benchmark results from the same configuration."""
        if len(results_list) == 1:
            return results_list[0]
        
        # Get all numeric keys that should be averaged
        numeric_keys = []
        for key, value in results_list[0].items():
            if isinstance(value, (int, float)) and key not in ['batch_size', 'input_len', 'output_len']:
                numeric_keys.append(key)
        
        # Create averaged result
        averaged = results_list[0].copy()
        
        for key in numeric_keys:
            values = [result[key] for result in results_list if key in result]
            if values:
                averaged[key] = float(np.mean(values))
                # Also store std dev for key metrics
                if key in ['prefill_latency', 'median_decode_latency', 'prefill_throughput', 'median_decode_throughput']:
                    averaged[f'{key}_std'] = float(np.std(values))
        
        # Store run statistics
        averaged['num_runs'] = len(results_list)
        averaged['runs_data'] = results_list
        
        return averaged
    
    def _run_comprehensive_sweep_with_env(
        self,
        model_path: str,
        max_batch_size: int,
        max_input_tokens: int,
        output_len: int,
        server_args: Dict[str, Any],
        use_cache: bool,
        num_runs: int,
        tp_rank: int,
        env: Dict[str, str]
    ):
        """Wrapper to set environment variables before running the sweep."""
        # Update environment variables for this process
        for key, value in env.items():
            os.environ[key] = value
        
        return self._run_comprehensive_sweep_single_process(
            model_path=model_path,
            max_batch_size=max_batch_size,
            max_input_tokens=max_input_tokens,
            output_len=output_len,
            server_args=server_args,
            use_cache=use_cache,
            num_runs=num_runs,
            tp_rank=tp_rank
        )
    
    def _run_comprehensive_sweep_single_process(
        self,
        model_path: str,
        max_batch_size: int,
        max_input_tokens: int,
        output_len: int,
        server_args: Dict[str, Any],
        use_cache: bool,
        num_runs: int,
        tp_rank: int = 0
    ) -> Dict[str, Any]:
        """Run comprehensive sweep in a single process (supports tensor parallelism)."""
        # Load model if needed
        server_args['cuda_graph_max_bs'] = max_batch_size
        self._load_model_if_needed(model_path, server_args, tp_rank)
        
        rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None
        
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
        
        rank_print(f"Batch sizes to test: {batch_sizes}")
        rank_print(f"Input lengths to test: {input_lens}")
        
        # Generate cache key
        cache_data = {
            'model_path': model_path,
            'batch_sizes': batch_sizes,
            'input_lens': input_lens,
            'output_len': output_len,
            'server_args': server_args,
            'benchmark_type': 'comprehensive_sweep',
            'num_runs': num_runs,
            'tp_size': server_args.get('tp_size', 1),
            'pp_size': server_args.get('pp_size', 1)
        }
        cache_key = self._get_cache_key(cache_data)
        
        # Try to load from cache (only on rank 0)
        cached_results = []
        cached_failed_configs = []
        completed_configs = set()
        
        if use_cache and tp_rank == 0:
            cached_data = self._load_cached_data(cache_key)
            if cached_data is not None:
                cached_results = cached_data.get('results', [])
                cached_metadata = cached_data.get('metadata', {})
                cached_failed_configs = cached_metadata.get('failed_configs', [])
                
                # Track which configs have been completed (either successfully or failed)
                for result in cached_results:
                    completed_configs.add((result['batch_size'], result['input_len']))
                for failed_config in cached_failed_configs:
                    completed_configs.add((failed_config['batch_size'], failed_config['input_len']))
                
                expected_total = len(batch_sizes) * len(input_lens)
                total_completed = len(completed_configs)
                
                if total_completed >= expected_total:
                    rank_print(f"Using complete cached results ({total_completed}/{expected_total} configs)")
                    return {
                        "metadata": cached_metadata,
                        "results": cached_results
                    }
                else:
                    rank_print(f"Found partial cached results ({total_completed}/{expected_total} configs), continuing from where left off")
        
        # Run benchmarks with error handling
        results = cached_results.copy()  # Start with cached results
        failed_configs = cached_failed_configs.copy()  # Start with cached failed configs
        
        total_configs = len(batch_sizes) * len(input_lens)
        current_config = 0
        
        for batch_size in batch_sizes:
            for input_len in input_lens:
                current_config += 1
                
                # Skip if this config has already been completed
                if (batch_size, input_len) in completed_configs:
                    continue
                    
                rank_print(f"Running config {current_config}/{total_configs}: batch_size={batch_size}, input_len={input_len}, output_len={output_len}")
                
                try:
                    # Run multiple times if requested
                    run_results = []
                    for run_idx in range(num_runs):
                        if num_runs > 1:
                            rank_print(f"    Run {run_idx + 1}/{num_runs}")
                        
                        reqs = prepare_synthetic_inputs_for_latency_test(batch_size, input_len)
                        result = latency_test_run_once(
                            run_name=f"sweep_bs{batch_size}_il{input_len}_run{run_idx}",
                            model_runner=self.model_runner,
                            rank_print=rank_print,
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
                            run_results.append(result)
                        else:
                            break  # If one run fails, stop trying more runs
                    
                    if run_results:
                        # Average the results if multiple runs
                        averaged_result = self._average_results(run_results)
                        results.append(averaged_result)
                        
                        decode_info = ""
                        if 'median_decode_latency' in averaged_result:
                            decode_info = f", decode={averaged_result['median_decode_latency']:.4f}s"
                            if f'median_decode_latency_std' in averaged_result and num_runs > 1:
                                decode_info += f"±{averaged_result['median_decode_latency_std']:.4f}s"
                            decode_info += f", decode_throughput={averaged_result['median_decode_throughput']:.1f} tok/s"
                        kv_cache_info = ""
                        if 'predicted_kv_usage_in_gb' in averaged_result:
                            kv_cache_info = f", kv_cache={averaged_result['predicted_kv_usage_in_gb']:.3f}GB"
                        
                        runs_info = f" (averaged over {len(run_results)} runs)" if num_runs > 1 else ""
                        rank_print(f"  ✓ Success: prefill={averaged_result['prefill_latency']:.4f}s, prefill_throughput={averaged_result['prefill_throughput']:.1f} tok/s{decode_info}{kv_cache_info}{runs_info}")
                    else:
                        failed_configs.append({'batch_size': batch_size, 'input_len': input_len, 'reason': 'skipped_due_to_limits'})
                        rank_print(f"  ⚠ Skipped due to resource limits")
                        
                except Exception as e:
                    failed_configs.append({'batch_size': batch_size, 'input_len': input_len, 'reason': str(e)})
                    rank_print(f"  ✗ Failed: {str(e)}")
                    continue
        
        # Count how many configs we just ran vs used from cache
        new_results = len(results) - len(cached_results)
        new_failed = len(failed_configs) - len(cached_failed_configs)
        
        rank_print(f"\nCompleted sweep: {len(results)} successful ({len(cached_results)} cached + {new_results} new), {len(failed_configs)} failed ({len(cached_failed_configs)} cached + {new_failed} new)")
        if failed_configs:
            rank_print("Failed configurations:")
            for config in failed_configs:
                rank_print(f"  bs={config['batch_size']}, il={config['input_len']}: {config['reason']}")
        
        metadata = {
            'benchmark_type': 'comprehensive_sweep',
            'model_path': model_path,
            'max_batch_size': max_batch_size,
            'max_input_tokens': max_input_tokens,
            'output_len': output_len,
            'batch_sizes_tested': batch_sizes,
            'input_lens_tested': input_lens,
            'successful_configs': len(results),
            'failed_configs': failed_configs,
            "tensor_parallel_size": server_args.get('tp_size', 1),
            "pipeline_parallel_size": server_args.get('pp_size', 1)
        }
        # Save to cache (only on rank 0)
        if use_cache and tp_rank == 0:
            self._save_results_to_cache(cache_key, results, metadata)
        
        return {
            'metadata': metadata,
            'results': results,
        }
    
    def run_comprehensive_sweep(
        self,
        model_path: str,
        max_batch_size: int = 64,
        max_input_tokens: int = 16384,
        output_len: int = 32,
        server_args: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        num_runs: int = 1
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmark sweeping batch sizes and input lengths.
        
        Args:
            model_path: Path to the model
            max_batch_size: Maximum batch size to test (will test powers of 2 up to this)
            max_input_tokens: Maximum input length to test (will test powers of 2 up to this)
            output_len: Output sequence length
            server_args: Optional server arguments
            use_cache: Whether to use cached results
            num_runs: Number of times to run each configuration for averaging
        
        Returns:
            Dict containing metadata and benchmark results
        """
        if server_args is None:
            server_args = {}
        
        tp_size = server_args.get('tp_size', 1)
        
        # Initialize port_args needed for distributed setup
        server_args_obj = ServerArgs(model_path=model_path)
        for key, value in server_args.items():
            if hasattr(server_args_obj, key):
                setattr(server_args_obj, key, value)
        port_args = PortArgs.init_new(server_args_obj)
        
        if tp_size == 1:
            # Single process execution
            return self._run_comprehensive_sweep_single_process(
                model_path=model_path,
                max_batch_size=max_batch_size,
                max_input_tokens=max_input_tokens,
                output_len=output_len,
                server_args=server_args,
                use_cache=use_cache,
                num_runs=num_runs,
                tp_rank=0
            )
        else:
            # Multi-process execution for tensor parallelism
            print(f"Running with tensor parallelism (tp_size={tp_size})")
            
            # Set up environment variables for distributed training
            os.environ["WORLD_SIZE"] = str(tp_size)
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(port_args.nccl_port)
            
            # Start worker processes
            workers = []
            for tp_rank in range(tp_size):
                # Set rank-specific environment variables
                env = os.environ.copy()
                env["RANK"] = str(tp_rank)
                env["LOCAL_RANK"] = str(tp_rank)
                
                proc = multiprocessing.Process(
                    target=self._run_comprehensive_sweep_with_env,
                    args=(
                        model_path,
                        max_batch_size,
                        max_input_tokens,
                        output_len,
                        server_args,
                        use_cache,
                        num_runs,
                        tp_rank,
                        env
                    )
                )
                proc.start()
                workers.append(proc)
            
            # Wait for all workers to complete
            for proc in workers:
                proc.join()
            
            # Clean up distributed environment
            if tp_size > 1:
                try:
                    destroy_distributed_environment()
                except:
                    pass
                try:
                    kill_process_tree(os.getpid(), include_parent=False)
                except:
                    pass
            
            # Return empty dict for multi-process case (results handled internally)
            return {"status": "completed", "tp_size": tp_size}
    
    def run_variable_length_benchmark(
        self,
        model_path: str,
        input_ids_batches: List[List[List[int]]],
        output_lens: List[int],
        server_args: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        batch_names: Optional[List[str]] = None,
        num_runs: int = 1
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
            'benchmark_type': 'variable_length',
            'num_runs': num_runs
        }
        cache_key = self._get_cache_key(cache_data)
        
        # Try to load from cache
        if use_cache:
            cached_results = self._load_cached_data(cache_key)
            if cached_results is not None:
                print(f"Using cached results for variable length benchmark")
                return cached_results
        
        # Load model if needed
        max_batch_size = max(len(batch) for batch in input_ids_batches)
        server_args['cuda_graph_max_bs'] = max_batch_size
        tp_size = server_args.get('tp_size', 1)
        
        if tp_size > 1:
            print(f"Variable length benchmark with tensor parallelism (tp_size={tp_size}) not yet supported. Using tp_size=1.")
            server_args = server_args.copy()
            server_args['tp_size'] = 1
            
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
                    # Run multiple times if requested
                    run_results = []
                    for run_idx in range(num_runs):
                        if num_runs > 1:
                            print(f"    Run {run_idx + 1}/{num_runs}")
                        
                        # Prepare requests with custom input_ids
                        reqs = prepare_synthetic_inputs_for_latency_test(
                            batch_size=batch_size,
                            input_len=0,  # Not used when input_ids_list is provided
                            input_ids_list=input_ids_batch
                        )
                        
                        # Run benchmark
                        result = latency_test_run_once(
                            run_name=f"var_len_{batch_name}_run{run_idx}",
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
                            run_results.append(result)
                        else:
                            break  # If one run fails, stop trying more runs
                    
                    if run_results:
                        # Average the results if multiple runs
                        averaged_result = self._average_results(run_results)
                        averaged_result['batch_name'] = batch_name
                        averaged_result['input_lengths'] = input_lengths
                        averaged_result['length_distribution'] = {
                            'min': min(input_lengths),
                            'max': max(input_lengths),
                            'mean': float(np.mean(input_lengths)),
                            'std': float(np.std(input_lengths))
                        }
                        results.append(averaged_result)
                        
                        decode_info = ""
                        if 'median_decode_latency' in averaged_result:
                            decode_info = f", decode={averaged_result['median_decode_latency']:.4f}s"
                            if f'median_decode_latency_std' in averaged_result and num_runs > 1:
                                decode_info += f"±{averaged_result['median_decode_latency_std']:.4f}s"
                            decode_info += f", decode_throughput={averaged_result['median_decode_throughput']:.1f} tok/s"
                        kv_cache_info = ""
                        if 'predicted_kv_usage_in_gb' in averaged_result:
                            kv_cache_info = f", kv_cache={averaged_result['predicted_kv_usage_in_gb']:.3f}GB"
                        
                        runs_info = f" (averaged over {len(run_results)} runs)" if num_runs > 1 else ""
                        print(f"  ✓ Success: prefill={averaged_result['prefill_latency']:.4f}s, prefill_throughput={averaged_result['prefill_throughput']:.1f} tok/s{decode_info}{kv_cache_info}{runs_info}")
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
        use_cache: bool = True,
        num_runs: int = 1
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
        
        # Ensure tensor parallelism compatibility 
        if server_args is None:
            server_args = {}
        tp_size = server_args.get('tp_size', 1)
        if tp_size > 1:
            print(f"Skewed batch benchmark with tensor parallelism (tp_size={tp_size}) not yet supported. Using tp_size=1.")
            server_args = server_args.copy()
            server_args['tp_size'] = 1
            
        return self.run_variable_length_benchmark(
            model_path=model_path,
            input_ids_batches=input_ids_batches,
            output_lens=[output_len],
            server_args=server_args,
            use_cache=use_cache,
            batch_names=batch_names,
            num_runs=num_runs
        )
    
    def clear_cache(self):
        """Clear all cached results."""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
        print("Cache cleared")
    

# Example usage
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    runner = BatchBenchmarkRunner()
    
    # Comprehensive sweep with error handling
    print("Running comprehensive sweep...")
    model = "Qwen/Qwen3-4B"
    tensor_parallel_size = 2
    pipeline_parallel_size = 1
    results_dict = runner.run_comprehensive_sweep(
        model_path=model,
        max_batch_size=64,
        max_input_tokens=2**14,
        output_len=32,
        num_runs=1,
        server_args={"load_format": "dummy", "tp_size": tensor_parallel_size, "pp_size": pipeline_parallel_size}
    )
    
    print(f"\nCompleted {len(results_dict.get('results', []))} successful experiments")
    model = model.replace("/", "_").replace(" ", "_")
    TP = tensor_parallel_size
    PP = pipeline_parallel_size
    with open(f"benchmark_data_{model}_TP_{TP}_PP_{PP}.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    # # List cached results
    # print("\nCached results:")
    # for cache_info in runner.list_cached_results():
    #     print(f"  {cache_info['filename']}: {cache_info['num_results']} results")