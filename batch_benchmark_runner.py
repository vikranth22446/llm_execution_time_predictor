import hashlib
import json
import os
import time
import signal
import multiprocessing
import subprocess
from typing import Any, Dict, List, Optional

import numpy as np

from sglang_batch_latency import (
    load_model,
    prepare_synthetic_inputs_for_latency_test,
    latency_test_run_once,
    _set_envs_and_config,
)
from sglang.srt.server_args import ServerArgs, PortArgs
from sglang.srt.distributed.parallel_state import destroy_distributed_environment
from sglang.srt.utils import kill_process_tree


def _powers_of_two(upto: int) -> List[int]:
    out, p = [], 0
    while (1 << p) <= upto:
        out.append(1 << p)
        p += 1
    return out

def _hash_key(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()

def _extract_gpu_model(gpu_name: str) -> str:
    """Extract the core GPU model name from full GPU string."""
    gpu_name = gpu_name.upper()
    
    # Common GPU model patterns
    gpu_patterns = [
        r'A100',
        r'V100',
        r'H100',
        r'RTX\s*(\d+)',
        r'GTX\s*(\d+)',
        r'TESLA\s*([A-Z]\d+)',
        r'QUADRO\s*([A-Z]\d+)',
        r'T4',
        r'P100',
        r'K80',
        r'L4'
    ]
    
    import re
    for pattern in gpu_patterns:
        match = re.search(pattern, gpu_name)
        if match:
            if match.groups():
                # For patterns with groups (like RTX 4090)
                return match.group(0).replace(' ', '')
            else:
                # For simple patterns (like A100)
                return match.group(0)
    
    # Fallback: try to extract any alphanumeric sequence after NVIDIA
    nvidia_match = re.search(r'NVIDIA\s+([A-Z0-9\-]+)', gpu_name)
    if nvidia_match:
        return nvidia_match.group(1)
    
    # Last fallback: return cleaned original name
    return ''.join(c for c in gpu_name if c.isalnum())[:10]

def _get_gpu_info() -> Dict[str, Any]:
    """Get GPU information including type and count."""
    try:
        # Get GPU names
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        gpu_names = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        
        if gpu_names:
            # Get unique GPU types and their counts
            from collections import Counter
            gpu_counts = Counter(gpu_names)
            
            # Extract clean model name for the primary GPU
            clean_gpu_name = _extract_gpu_model(gpu_names[0])
            
            return {
                "gpu_type": gpu_names[0],  # Full GPU name for metadata
                "gpu_model": clean_gpu_name,  # Clean model name for filename
                "gpu_count": len(gpu_names),
                "gpu_distribution": dict(gpu_counts)
            }
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    return {
        "gpu_type": "unknown",
        "gpu_model": "unknown",
        "gpu_count": 0,
        "gpu_distribution": {}
    }

def _average_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results_list:
        return {}
    out = dict(results_list[0])
    numeric_keys = [
        k for k, v in results_list[0].items()
        if isinstance(v, (int, float)) and k not in ("batch_size", "input_len", "output_len")
    ]
    for k in numeric_keys:
        vals = [r[k] for r in results_list if k in r]
        if vals:
            out[k] = float(np.mean(vals))
    for k in ("prefill_latency", "median_decode_latency", "prefill_throughput", "median_decode_throughput"):
        vals = [r.get(k) for r in results_list if r.get(k) is not None]
        if vals:
            out[f"{k}_std"] = float(np.std(vals))
    out["num_runs"] = len(results_list)
    out["runs_data"] = results_list
    return out


class SimpleBenchmarkRunner:
    def __init__(self, cache_dir: str = "./benchmark_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.model_runner = None
        self.tokenizer = None
        self.server_args_obj = None
        self._loaded_sig = None  # (model_path, server_args_hash)

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def load_cache(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(key)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return None
    
    def _get_completed_configs(self, cached_data: Dict[str, Any]) -> set:
        """Extract completed configurations from cached data."""
        completed_configs = set()
        
        # Add successful configs
        for result in cached_data.get('results', []):
            completed_configs.add((result['batch_size'], result['input_len']))
        
        # Add failed configs
        metadata = cached_data.get('metadata', {})
        for failed_config in metadata.get('failed_configs', []):
            completed_configs.add((failed_config['batch_size'], failed_config['input_len']))
            
        return completed_configs

    def save_cache(self, key: str, data: Dict[str, Any]) -> None:
        path = self._cache_path(key)
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[cache] failed to save: {e}")

    def _load_model_if_needed(
        self,
        model_path: str,
        server_args: Dict[str, Any],
        tp_rank: int,
        nccl_port: int,
    ) -> None:
        sig = (model_path, _hash_key(server_args))
        if self._loaded_sig == sig:
            return
        server_args_obj = ServerArgs(model_path=model_path)
        for k, v in server_args.items():
            if hasattr(server_args_obj, k):
                setattr(server_args_obj, k, v)
        _set_envs_and_config(server_args_obj)
        port_args = PortArgs.init_new(server_args_obj)
        port_args.nccl_port = int(nccl_port)
        rank_print = print if tp_rank == 0 else (lambda *a, **k: None)
        rank_print(f"[Rank {tp_rank}] Loading {model_path} (tp_size={server_args.get('tp_size', 1)})")
        self.model_runner, self.tokenizer = load_model(server_args_obj, port_args, tp_rank)
        self.server_args_obj = server_args_obj
        self._loaded_sig = sig
        rank_print(f"[Rank {tp_rank}] Model ready")

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
    ) -> Dict[str, Any]:
        server_args = dict(server_args or {})
        server_args["cuda_graph_max_bs"] = max_batch_size
        self._load_model_if_needed(model_path, server_args, tp_rank, nccl_port)
        rank_print = print if tp_rank == 0 else (lambda *a, **k: None)

        batch_sizes = _powers_of_two(max_batch_size)
        input_lens = _powers_of_two(max_input_tokens)
        rank_print(f"[Rank {tp_rank}] Sweep: bs={batch_sizes}, il={input_lens}, ol={output_len}, runs={num_runs}")

        # Start with cached results if available
        results: List[Dict[str, Any]] = cached_results.copy() if cached_results else []
        failed_configs: List[Dict[str, Any]] = cached_failed_configs.copy() if cached_failed_configs else []
        completed_configs_set = completed_configs if completed_configs else set()
        total_trails = len(batch_sizes) * len(input_lens)
        completed_trials = 0
        for bs in batch_sizes:
            for il in input_lens:
                # Skip if this config has already been completed
                if (bs, il) in completed_configs_set:
                    continue
                    
                completed_trials += 1
                rank_print(f"[{completed_trials}/{total_trails}] Testing bs={bs}, il={il}, output_len={output_len}")
                run_results: List[Dict[str, Any]] = []
                failed_reason: Optional[str] = None
                for run_idx in range(num_runs):
                    if num_runs > 1:
                        rank_print(f"\tRunning bs={bs}, il={il}, run={run_idx + 1}/{num_runs}")
                    try:
                        reqs = prepare_synthetic_inputs_for_latency_test(bs, il)
                        ret = latency_test_run_once(
                            run_name=f"sweep_bs{bs}_il{il}_run{run_idx}",
                            model_runner=self.model_runner,
                            rank_print=lambda *a, **k: None,
                            reqs=reqs,
                            batch_size=bs,
                            input_len=il,
                            output_len=output_len,
                            device=self.server_args_obj.device,
                            log_decode_step=1000,
                            profile=False,
                            profile_filename_prefix="",
                        )
                        if ret is None:
                            failed_reason = "skipped_due_to_limits"
                            break
                        run_results.append(ret)
                    except RuntimeError as e:
                        msg = str(e).lower()
                        if "out of memory" in msg:
                            failed_reason = "oom"
                        else:
                            failed_reason = f"runtime_error: {e}"
                        break
                    except Exception as e:
                        failed_reason = f"exception: {e}"
                        break

                if run_results:
                    averaged = _average_results(run_results)
                    averaged["batch_size"] = bs
                    averaged["input_len"] = il
                    averaged["output_len"] = output_len
                    if failed_reason is not None and len(run_results) < num_runs:
                        averaged["num_failed_runs"] = num_runs - len(run_results)
                        averaged["partial_success_reason"] = failed_reason
                    results.append(averaged)
                else:
                    failed_configs.append({"batch_size": bs, "input_len": il, "output_len": output_len, "reason": failed_reason or "unknown"})

        gpu_info = _get_gpu_info()
        return {
            "metadata": {
                "benchmark_type": "comprehensive_sweep",
                "model_path": model_path,
                "max_batch_size": max_batch_size,
                "max_input_tokens": max_input_tokens,
                "output_len": output_len,
                "tp_size": server_args.get("tp_size", 1),
                "pp_size": server_args.get("pp_size", 1),
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

    def _worker_entry(
        self,
        model_path: str,
        max_batch_size: int,
        max_input_tokens: int,
        output_len: int,
        server_args: Dict[str, Any],
        tp_rank: int,
        env: Dict[str, str],
        nccl_port: int,
        num_runs: int,
    ) -> None:
        try:
            for k, v in env.items():
                os.environ[k] = v
            os.environ["MASTER_PORT"] = str(nccl_port)
            r = SimpleBenchmarkRunner(cache_dir=self.cache_dir)
            _ = r._sweep_single_process(
                model_path=model_path,
                max_batch_size=max_batch_size,
                max_input_tokens=max_input_tokens,
                output_len=output_len,
                server_args=server_args,
                tp_rank=tp_rank,
                nccl_port=nccl_port,
                num_runs=num_runs,
                cached_results=[],
                cached_failed_configs=[],
                completed_configs=set(),
            )
        finally:
            try:
                destroy_distributed_environment()
            except Exception:
                pass

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
    ) -> Dict[str, Any]:
        server_args = dict(server_args or {})
        tp_size = int(server_args.get("tp_size", 1))

        sargs = ServerArgs(model_path=model_path)
        for k, v in server_args.items():
            if hasattr(sargs, k):
                setattr(sargs, k, v)
        pargs = PortArgs.init_new(sargs)
        nccl_port = pargs.nccl_port

        os.environ["WORLD_SIZE"] = str(tp_size)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(nccl_port)

        cache_key = _hash_key(
            dict(
                model_path=model_path,
                max_batch_size=max_batch_size,
                max_input_tokens=max_input_tokens,
                output_len=output_len,
                server_args=server_args,
                tp_size=tp_size,
                tag=cache_tag,
                num_runs=num_runs,
            )
        )
        # Handle caching logic
        cached_results = []
        cached_failed_configs = []
        completed_configs = set()
        
        if use_cache and not skip_cache and tp_size == 1:
            cached = self.load_cache(cache_key)
            if cached:
                # Check if we have complete results
                cached_results = cached.get('results', [])
                cached_metadata = cached.get('metadata', {})
                cached_failed_configs = cached_metadata.get('failed_configs', [])
                completed_configs = self._get_completed_configs(cached)
                
                # Calculate expected total configurations
                batch_sizes = _powers_of_two(max_batch_size)
                input_lens = _powers_of_two(max_input_tokens)
                expected_total = len(batch_sizes) * len(input_lens)
                total_completed = len(completed_configs)
                
                if total_completed >= expected_total:
                    print(f"[cache] Using complete cached results ({total_completed}/{expected_total} configs)")
                    return cached
                else:
                    print(f"[cache] Found partial cached results ({total_completed}/{expected_total} configs), continuing from where left off")

        if tp_size == 1:
            print(f"[tp] single process, runs={num_runs}")
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
            )
            if use_cache:
                self.save_cache(cache_key, out)
            return out

        print(f"[tp] tensor parallel = {tp_size}, port={nccl_port}, runs={num_runs}")
        ctx = multiprocessing.get_context("spawn")
        children: List[multiprocessing.Process] = []
        for rank in range(1, tp_size):
            env = os.environ.copy()
            env["RANK"] = str(rank)
            env["LOCAL_RANK"] = str(rank)
            p = ctx.Process(
                target=self._worker_entry,
                args=(
                    model_path,
                    max_batch_size,
                    max_input_tokens,
                    output_len,
                    server_args,
                    rank,
                    env,
                    nccl_port,
                    num_runs,
                ),
            )
            p.start()
            children.append(p)

        env0 = os.environ.copy()
        env0["RANK"] = "0"
        env0["LOCAL_RANK"] = "0"
        for k, v in env0.items():
            os.environ[k] = v

        out = None
        try:
            out = self._sweep_single_process(
                model_path=model_path,
                max_batch_size=max_batch_size,
                max_input_tokens=max_input_tokens,
                output_len=output_len,
                server_args=server_args,
                tp_rank=0,
                nccl_port=nccl_port,
                num_runs=num_runs,
                cached_results=[],
                cached_failed_configs=[],
                completed_configs=set(),
            )
        finally:
            try:
                destroy_distributed_environment()
            except Exception:
                pass

        deadline = time.time() + 3600
        for p in children:
            remain = max(0, deadline - time.time())
            p.join(timeout=remain)
            if p.is_alive():
                p.terminate()
                p.join(5)
                if p.is_alive():
                    try:
                        os.kill(p.pid, signal.SIGKILL)
                    except Exception:
                        pass

        if out is None:
            return {"status": "completed_with_warnings", "message": "no result from rank 0"}

        if use_cache:
            self.save_cache(cache_key, out)
        return out

def profile():
    try:
        multiprocessing.set_start_method("spawn", force=True)
        print("[mp] start method = spawn")
    except RuntimeError as e:
        print(f"[mp] start method already set: {e}")

    runner = SimpleBenchmarkRunner()

    model = "Qwen/Qwen3-4B"
    tp = 1
    pp = 1

    try:
        results = runner.run_sweep(
            model_path=model,
            max_batch_size=64,
            max_input_tokens=2**14,
            output_len=32,
            server_args={"load_format": "dummy", "tp_size": tp, "pp_size": pp},
            use_cache=True,
            skip_cache=False,  # Set to True to skip cached configs and start fresh
            cache_tag="v1",
            num_runs=3,
        )
        print(f"\n[done] {len(results.get('results', []))} successful configs")
        print(f"[failed] {len(results.get('metadata', {}).get('failed_configs', []))} failed configs")
        
        # Get GPU model name for filename
        gpu_model = results.get('metadata', {}).get('gpu_model', 'unknown')
        
        out_name = f"benchmark_data_{model.replace('/','_')}_TP_{tp}_PP_{pp}_{gpu_model}.json"
        with open(out_name, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[save] {out_name}")
    except Exception as e:
        print(f"[error] {e}")
        import traceback; traceback.print_exc()
    finally:
        try:
            kill_process_tree(os.getpid(), include_parent=False)
        except Exception:
            pass

if __name__ == "__main__":
    profile()