from __future__ import annotations

import json
import os
import time
import multiprocessing
import torch
import gc
from typing import Any, Dict, List, Optional, Set, Tuple

from .bench_backend_handler import Backend, SGLangBackend, VLLMBackend
from .bench_utils import (powers_of_two, hash_key, get_gpu_info, average_results, 
                         load_json_file, save_json_file, create_backend, format_output_filename)

class SimpleBenchmarkRunner:
    def __init__(self, backend: Backend, cache_dir: str = "./benchmark_cache"):
        self.backend = backend
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def load_cache(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(key)
        return load_json_file(path) if os.path.exists(path) else None

    def save_cache(self, key: str, data: Dict[str, Any]) -> None:
        path = self._cache_path(key)
        if not save_json_file(path, data):
            print(f"[cache] failed to save: {path}")

    @staticmethod
    def _get_completed_configs(cached_data: Dict[str, Any]) -> Set[Tuple[int, int]]:
        done = set()
        for r in cached_data.get("results", []):
            done.add((r["batch_size"], r["input_len"]))
        for fc in cached_data.get("metadata", {}).get("failed_configs", []):
            done.add((fc["batch_size"], fc["input_len"]))
        return done
    
    def _run_single_inference(self, model_path: str, server_args: Dict[str, Any], 
                             nccl_port: int, batch_size: int, input_len: int, 
                             output_len: int, run_name: str, sandbox: bool,
                             chunk_prefill: bool = False, chunk_size: int = 512) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Run single inference - returns (result, error)"""
        try:
            if sandbox:
                from .sandbox_runner import run_sandboxed_inference
                return run_sandboxed_inference(
                    model_path, server_args, self.backend.name, nccl_port,
                    batch_size, input_len, output_len, run_name, 200.0, chunk_prefill, chunk_size
                )
            else:
                reqs = self.backend.prepare_inputs(batch_size, input_len)
                result = self.backend.run_once(run_name, reqs, batch_size, input_len, output_len, chunk_prefill, chunk_size)
                return result, "skipped_due_to_limits" if result is None else None
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                gc.collect()
                return None, "oom"
            return None, f"runtime_error: {e}"
        except Exception as e:
            return None, f"exception: {e}"

    def _sweep_single_process(
        self,
        model_path: str,
        max_batch_size: int,
        max_input_tokens: int,
        output_len: int,
        server_args: Dict[str, Any],
        tp_rank: int,
        nccl_port: int,
        num_runs: int,
        cached_results: Optional[List[Dict[str, Any]]] = None,
        cached_failed_configs: Optional[List[Dict[str, Any]]] = None,
        completed_configs: Optional[set] = None,
        cache_key: Optional[str] = None,
        use_cache: bool = True,
        sandbox: bool = False,
        chunk_prefill: bool = False,
        chunk_size: int = 512,
        max_input_tokens_start_chunking: int = 100000,
    ) -> Dict[str, Any]:
        server_args = dict(server_args or {})
        server_args["cuda_graph_max_bs"] = max_batch_size
        server_args.setdefault("model_path", model_path)

        # Load model only if not in sandbox mode
        if not sandbox:
            self.backend.load_model_if_needed(model_path, server_args, tp_rank, nccl_port)
        all_batch_sizes_consider = [1,2, 4, 8, 16, 32, 48, 64]
        input_lens = [1,2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 12000, 16384]
        combinations = [(bs, il) for bs in all_batch_sizes_consider for il in input_lens]
        combinations.append((128, 256))
        combinations.append((128, 512))
        combinations = [
            (bs, il) for bs, il in combinations
            if bs <= max_batch_size and il <= max_input_tokens
        ]
        batch_sizes = list(sorted(set(bs for bs, _ in combinations)))

        results: List[Dict[str, Any]] = list(cached_results or [])
        failed_configs: List[Dict[str, Any]] = list(cached_failed_configs or [])
        completed = set(completed_configs or set())

        total = len(combinations) - len(completed_configs)
        done = 0
        start_time = time.time()
        
        # For ETA calculation
        config_times = []

        for bs, il in combinations:
            if (bs, il) in completed:
                continue
            
            config_start_time = time.time()
            done += 1
            
            # Calculate ETA
            if config_times:
                avg_time_per_config = sum(config_times) / len(config_times)
                remaining_configs = total - done + 1
                eta_seconds = avg_time_per_config * remaining_configs
                eta_minutes = eta_seconds / 60
                if eta_minutes < 60:
                    eta_str = f"ETA: {eta_minutes:.1f}m"
                else:
                    eta_hours = eta_minutes / 60
                    eta_str = f"ETA: {eta_hours:.1f}h"
            else:
                eta_str = "ETA: calculating..."
            
            print(f"[{done}/{total}] bs={bs} il={il} ol={output_len} | {eta_str}")

            run_results: List[Dict[str, Any]] = []
            failed_reason: Optional[str] = None

            for run_idx in range(num_runs):
                run_name = f"sweep_bs{bs}_il{il}_run{run_idx}"

                # To prevent errors, when there's a lot of input tokens, we can auto-chunk
                auto_chunk_prefill = chunk_prefill or (bs * il >= max_input_tokens_start_chunking)
                auto_chunk_size = chunk_size if chunk_prefill else 512
                
                ret, error = self._run_single_inference(
                    model_path, server_args, nccl_port, bs, il, output_len, run_name, sandbox, auto_chunk_prefill, auto_chunk_size
                )
                if ret is None:
                    failed_reason = "oom" if error == "oom" else error or "unknown"
                    break
                run_results.append(ret)

            if run_results:
                avg = average_results(run_results)
                avg["batch_size"] = bs
                avg["input_len"] = il
                avg["output_len"] = output_len
                if failed_reason and len(run_results) < num_runs:
                    avg["num_failed_runs"] = num_runs - len(run_results)
                    avg["partial_success_reason"] = failed_reason
                results.append(avg)
            else:
                failed_configs.append({
                    "batch_size": bs, "input_len": il, "output_len": output_len,
                    "reason": failed_reason or "unknown"
                })
            
            # Record time taken for this configuration
            config_end_time = time.time()
            config_duration = config_end_time - config_start_time
            config_times.append(config_duration)
            
            # Keep only last 10 measurements for moving average
            if len(config_times) > 10:
                config_times = config_times[-10:]

            # Save cache incrementally after each configuration (only on rank 0)
            if use_cache and cache_key and tp_rank == 0:
                gpu_info = get_gpu_info()
                current_result = {
                    "metadata": {
                        "benchmark_type": "comprehensive_sweep",
                        "model_path": model_path,
                        "backend": self.backend.name,
                        "max_batch_size": max_batch_size,
                        "max_input_tokens": max_input_tokens,
                        "output_len": output_len,
                        "tp_size": int(server_args.get("tp_size", 1)),
                        "pp_size": int(server_args.get("pp_size", 1)) if "pp_size" in server_args else 1,
                        "batch_sizes_tested": batch_sizes,
                        "input_lens_tested": input_lens,
                        "num_runs": num_runs,
                        "failed_configs": failed_configs,
                        "successful_configs": len(results),
                        "gpu_type": gpu_info["gpu_type"],
                        "gpu_model": gpu_info["gpu_model"],
                        "gpu_count": gpu_info["gpu_count"],
                        "gpu_distribution": gpu_info["gpu_distribution"],
                    },
                    "results": results,
                }
                self.save_cache(cache_key, current_result)

        gpu_info = get_gpu_info()
        return {
            "metadata": {
                "benchmark_type": "comprehensive_sweep",
                "model_path": model_path,
                "backend": self.backend.name,
                "max_batch_size": max_batch_size,
                "max_input_tokens": max_input_tokens,
                "output_len": output_len,
                "tp_size": int(server_args.get("tp_size", 1)),
                "pp_size": int(server_args.get("pp_size", 1)) if "pp_size" in server_args else 1,
                "batch_sizes_tested": batch_sizes,
                "input_lens_tested": input_lens,
                "num_runs": num_runs,
                "failed_configs": failed_configs,
                "successful_configs": len(results),
                "gpu_type": gpu_info["gpu_type"],
                "gpu_model": gpu_info["gpu_model"],
                "gpu_count": gpu_info["gpu_count"],
                "gpu_distribution": gpu_info["gpu_distribution"],
            },
            "results": results,
        }


    def run_sweep(
        self,
        model_path: str,
        max_batch_size: int = 64,
        max_input_tokens: int = 16384,
        output_len: int = 32,
        server_args: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        skip_cache: bool = False,
        cache_tag: Optional[str] = None,
        num_runs: int = 1,
        sandbox: bool = False,
        chunk_prefill: bool = False,
        chunk_size: int = 512,
        max_input_tokens_start_chunking: int = 100000,
    ) -> Dict[str, Any]:
        server_args = dict(server_args or {})
        tp_size = int(server_args.get("tp_size", 1))
        server_args["model_path"] = model_path

        nccl_port = self.backend.init_env_and_nccl(server_args)
        cache_key = hash_key({
            "model_path": model_path,
            "max_batch_size": max_batch_size,
            "max_input_tokens": max_input_tokens,
            "output_len": output_len,
            "server_args": server_args,
            "tp_size": tp_size,
            "tag": cache_tag,
            "num_runs": num_runs,
            "backend": self.backend.name,
        })

        cached_results: List[Dict[str, Any]] = []
        cached_failed_configs: List[Dict[str, Any]] = []
        completed_configs = set()

        if use_cache and not skip_cache and (not self.backend.supports_external_tp or tp_size == 1):
            cached = self.load_cache(cache_key)
            if cached:
                cached_results = cached.get("results", [])
                meta = cached.get("metadata", {})
                cached_failed_configs = meta.get("failed_configs", [])
                completed_configs = self._get_completed_configs(cached)

                bs_list = powers_of_two(max_batch_size)
                il_list = powers_of_two(max_input_tokens)
                expected = len(bs_list) * len(il_list)
                if len(completed_configs) >= expected:
                    print(f"[cache] Using complete cached results ({len(completed_configs)}/{expected}) in file {self._cache_path(cache_key)}")
                    return cached
                else:
                    print(f"[cache] Partial cached results ({len(completed_configs)}/{expected}); resuming")

        # Single process path (vLLM always, or sglang with sandbox mode)
        if self.backend.name == "vllm" or sandbox:
            mode_str = "sandbox" if sandbox else "single-process"
            print(f"[{self.backend.name}] {mode_str} runs={num_runs}")
            out = self._sweep_single_process(
                model_path=model_path,
                max_batch_size=max_batch_size,
                max_input_tokens=max_input_tokens,
                output_len=output_len,
                server_args=server_args,
                tp_rank=0,
                nccl_port=nccl_port,
                num_runs=num_runs,
                cached_results=cached_results,
                cached_failed_configs=cached_failed_configs,
                completed_configs=completed_configs,
                cache_key=cache_key,
                use_cache=use_cache,
                sandbox=sandbox,
                chunk_prefill=chunk_prefill,
                chunk_size=chunk_size,
                max_input_tokens_start_chunking=max_input_tokens_start_chunking,
            )
            if use_cache:
                self.save_cache(cache_key, out)
            return out

        # TP is now handled internally by the backend
        print(f"[{self.backend.name}] tensor-parallel={tp_size} port={nccl_port} runs={num_runs}")
        out = self._sweep_single_process(
            model_path=model_path,
            max_batch_size=max_batch_size,
            max_input_tokens=max_input_tokens,
            output_len=output_len,
            server_args=server_args,
            tp_rank=0,
            nccl_port=nccl_port,
            num_runs=num_runs,
            cached_results=cached_results,
            cached_failed_configs=cached_failed_configs,
            completed_configs=completed_configs,
            cache_key=cache_key,
            use_cache=use_cache,
            sandbox=False,
            chunk_prefill=chunk_prefill,
            chunk_size=chunk_size,
            max_input_tokens_start_chunking=max_input_tokens_start_chunking,
        )
        if use_cache:
            self.save_cache(cache_key, out)
        return out

def profile(backend_name: str = "sglang", sandbox: bool = False) -> None:
    if backend_name == "sglang":
        try:
            multiprocessing.set_start_method("spawn", force=True)
            print("[mp] start method = spawn")
        except RuntimeError as e:
            print(f"[mp] start method already set: {e}")

    backend = create_backend(backend_name)
    runner = SimpleBenchmarkRunner(backend=backend)

    model = "Qwen/Qwen3-4B"
    tp, pp = 1, 1

    try:
        server_args = {"load_format": "dummy", "tp_size": tp}
        if backend_name == "sglang":
            server_args["pp_size"] = pp

        results = runner.run_sweep(
            model_path=model,
            max_batch_size=64,
            max_input_tokens=2**14,
            output_len=32,
            server_args=server_args,
            use_cache=True,
            skip_cache=False,
            cache_tag="v1",
            num_runs=3,
            sandbox=sandbox,
        )
        print(f"\n[done] {len(results.get('results', []))} successful configs")
        print(f"[failed] {len(results.get('metadata', {}).get('failed_configs', []))} failed configs")

        gpu_model = results.get("metadata", {}).get("gpu_model", "unknown")
        out_name = format_output_filename(model, tp, backend_name, gpu_model, pp)
        
        if save_json_file(out_name, results):
            print(f"[save] {out_name}")
        else:
            print(f"[error] Failed to save {out_name}")
    except Exception as e:
        print(f"[error] {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["sglang", "vllm"], default="sglang", 
                       help="Backend to use for benchmarking")
    parser.add_argument("--sandbox", action="store_true",
                       help="Run each inference in isolated process for OOM resilience")
    args = parser.parse_args()
    
    profile(backend_name=args.backend, sandbox=args.sandbox)