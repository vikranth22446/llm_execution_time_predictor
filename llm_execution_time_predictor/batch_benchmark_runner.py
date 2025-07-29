from __future__ import annotations

import json
import os
import signal
import time
import multiprocessing
from typing import Any, Dict, List, Optional, Set, Tuple

from .bench_backend_handler import Backend, SGLangBackend, VLLMBackend
from .bench_utils import powers_of_two, hash_key, get_gpu_info, average_results

class SimpleBenchmarkRunner:
    def __init__(self, backend: Backend, cache_dir: str = "./benchmark_cache"):
        self.backend = backend
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

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

    def save_cache(self, key: str, data: Dict[str, Any]) -> None:
        path = self._cache_path(key)
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[cache] failed to save: {e}")

    @staticmethod
    def _get_completed_configs(cached_data: Dict[str, Any]) -> Set[Tuple[int, int]]:
        done = set()
        for r in cached_data.get("results", []):
            done.add((r["batch_size"], r["input_len"]))
        for fc in cached_data.get("metadata", {}).get("failed_configs", []):
            done.add((fc["batch_size"], fc["input_len"]))
        return done

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

        # make model_path available for backends that build ServerArgs
        server_args.setdefault("model_path", model_path)

        self.backend.load_model_if_needed(model_path, server_args, tp_rank, nccl_port)

        batch_sizes = powers_of_two(max_batch_size)
        input_lens = powers_of_two(max_input_tokens)

        results: List[Dict[str, Any]] = list(cached_results or [])
        failed_configs: List[Dict[str, Any]] = list(cached_failed_configs or [])
        completed = set(completed_configs or set())

        total = len(batch_sizes) * len(input_lens)
        done = 0

        for bs in batch_sizes:
            for il in input_lens:
                if (bs, il) in completed:
                    continue
                done += 1
                print(f"[{done}/{total}] bs={bs} il={il} ol={output_len}")

                run_results: List[Dict[str, Any]] = []
                failed_reason: Optional[str] = None

                for run_idx in range(num_runs):
                    try:
                        reqs = self.backend.prepare_inputs(bs, il)
                        ret = self.backend.run_once(
                            run_name=f"sweep_bs{bs}_il{il}_run{run_idx}",
                            reqs=reqs,
                            batch_size=bs,
                            input_len=il,
                            output_len=output_len,
                        )
                        if ret is None:
                            failed_reason = "skipped_due_to_limits"
                            break
                        run_results.append(ret)
                    except RuntimeError as e:
                        msg = str(e).lower()
                        failed_reason = "oom" if "out of memory" in msg else f"runtime_error: {e}"
                        break
                    except Exception as e:
                        failed_reason = f"exception: {e}"
                        break

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

            backend = SGLangBackend() if self.backend.name == "sglang" else VLLMBackend()
            child = SimpleBenchmarkRunner(backend=backend, cache_dir=self.cache_dir)
            _ = child._sweep_single_process(
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
            self.backend.destroy()

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

        # Single process path (vLLM always, or sglang with tp_size=1)
        if not self.backend.supports_external_tp or tp_size == 1:
            print(f"[{self.backend.name}] single-process runs={num_runs}")
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

        # Multi-process TP path (sglang)
        print(f"[{self.backend.name}] tensor-parallel={tp_size} port={nccl_port} runs={num_runs}")
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
                cached_results=cached_results,
                cached_failed_configs=cached_failed_configs,
                completed_configs=completed_configs,
            )
        finally:
            self.backend.destroy()

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

def profile(backend_name: str = "sglang") -> None:
    if backend_name == "sglang":
        try:
            multiprocessing.set_start_method("spawn", force=True)
            print("[mp] start method = spawn")
        except RuntimeError as e:
            print(f"[mp] start method already set: {e}")

    backend = SGLangBackend() if backend_name == "sglang" else VLLMBackend()
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
        )
        print(f"\n[done] {len(results.get('results', []))} successful configs")
        print(f"[failed] {len(results.get('metadata', {}).get('failed_configs', []))} failed configs")

        gpu_model = results.get("metadata", {}).get("gpu_model", "unknown")
        if backend_name == "sglang":
            out_name = f"benchmark_data_{model.replace('/','_')}_TP_{tp}_PP_{pp}_{gpu_model}.json"
        else:
            out_name = f"benchmark_data_{model.replace('/','_')}_TP_{tp}_{backend_name}_{gpu_model}.json"

        with open(out_name, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[save] {out_name}")
    except Exception as e:
        print(f"[error] {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["sglang", "vllm"], default="sglang", 
                       help="Backend to use for benchmarking")
    args = parser.parse_args()
    
    profile(backend=args.backend)