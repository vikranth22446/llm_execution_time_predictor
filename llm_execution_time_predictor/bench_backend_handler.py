from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import time

class Backend(ABC):
    name: str

    @property
    def supports_external_tp(self) -> bool:
        """True if TP is handled by multiple Python processes (e.g., sglang)."""
        return False

    @abstractmethod
    def init_env_and_nccl(self, server_args: Dict[str, Any]) -> int:
        """Return NCCL port to use (0 if not applicable)."""
        raise NotImplementedError

    @abstractmethod
    def load_model_if_needed(
        self,
        model_path: str,
        server_args: Dict[str, Any],
        tp_rank: int,
        nccl_port: int,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def prepare_inputs(self, batch_size: int, input_len: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def run_once(
        self,
        run_name: str,
        reqs: Any,
        batch_size: int,
        input_len: int,
        output_len: int,
        chunk_prefill: bool = False,
        chunk_size: int = 512,
    ) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def destroy(self) -> None:
        pass


class SGLangBackend(Backend):
    name = "sglang"

    def __init__(self) -> None:
        self._model_runner = None
        self._tokenizer = None
        self._server_args_obj = None
        self._loaded_sig: Optional[Tuple[str, str]] = None  # (model_path, server_args_hash)
        self._tp_workers: List[Any] = []

    @property
    def supports_external_tp(self) -> bool:
        return False  # Now handled internally

    def init_env_and_nccl(self, server_args: Dict[str, Any]) -> int:
        from sglang.srt.server_args import ServerArgs, PortArgs

        tp_size = int(server_args.get("tp_size", 1))
        sargs = ServerArgs(model_path=server_args.get("model_path", ""))
        for k, v in server_args.items():
            if hasattr(sargs, k):
                setattr(sargs, k, v)
        pargs = PortArgs.init_new(sargs)

        os.environ["WORLD_SIZE"] = str(tp_size)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(pargs.nccl_port)
        return int(pargs.nccl_port)

    def load_model_if_needed(
        self,
        model_path: str,
        server_args: Dict[str, Any],
        tp_rank: int,
        nccl_port: int,
    ) -> None:
        from .sglang_batch_latency import load_model, _set_envs_and_config
        from sglang.srt.server_args import ServerArgs, PortArgs
        from .bench_utils import hash_key

        sig = (model_path, hash_key(server_args))
        if self._loaded_sig == sig:
            return

        server_args_obj = ServerArgs(model_path=model_path)
        for k, v in server_args.items():
            if hasattr(server_args_obj, k):
                setattr(server_args_obj, k, v)
        
        tp_size = int(server_args.get("tp_size", 1))
        
        if tp_size > 1:
            self._load_with_tp(server_args_obj, tp_size, nccl_port)
        else:
            _set_envs_and_config(server_args_obj)
            port_args = PortArgs.init_new(server_args_obj)
            port_args.nccl_port = int(nccl_port)
            self._model_runner, self._tokenizer = load_model(server_args_obj, port_args, tp_rank)
        
        self._server_args_obj = server_args_obj
        self._loaded_sig = sig

    def prepare_inputs(self, batch_size: int, input_len: int) -> Any:
        from .sglang_batch_latency import prepare_synthetic_inputs_for_latency_test
        return prepare_synthetic_inputs_for_latency_test(batch_size, input_len)

    def run_once(
        self,
        run_name: str,
        reqs: Any,
        batch_size: int,
        input_len: int,
        output_len: int,
        chunk_prefill: bool = False,
        chunk_size: int = 512,
    ) -> Optional[Dict[str, Any]]:
        from .sglang_batch_latency import latency_test_run_once
        return latency_test_run_once(
            run_name=run_name,
            model_runner=self._model_runner,
            rank_print=lambda *a, **k: None,
            reqs=reqs,
            batch_size=batch_size,
            input_len=input_len,
            output_len=output_len,
            device=self._server_args_obj.device if self._server_args_obj else "cuda",
            log_decode_step=1000,
            profile=False,
            profile_filename_prefix="",
            chunk_prefill=chunk_prefill,
            chunk_size=chunk_size,
        )

    def _load_with_tp(self, server_args_obj, tp_size: int, nccl_port: int):
        """Load model with tensor parallelism using multiple processes"""
        import multiprocessing
        import time
        for rank in range(1, tp_size):
            env = os.environ.copy()
            env["RANK"] = str(rank)
            env["LOCAL_RANK"] = str(rank)
            env["MASTER_PORT"] = str(nccl_port)
            
            # Convert ServerArgs object to dict to avoid pickle issues with weakrefs
            server_args_dict = {
                'model_path': server_args_obj.model_path,
                'tp_size': server_args_obj.tp_size,
                'device': getattr(server_args_obj, 'device', 'cuda'),
                'load_format': getattr(server_args_obj, 'load_format', 'auto'),
                'trust_remote_code': getattr(server_args_obj, 'trust_remote_code', True),
                'max_prefill_tokens': getattr(server_args_obj, 'max_prefill_tokens', 16384),
                'chunked_prefill_size': getattr(server_args_obj, 'chunked_prefill_size', -1),
                'enable_mixed_chunk': getattr(server_args_obj, 'enable_mixed_chunk', False),
                'max_prefill_tokens_per_chunk': getattr(server_args_obj, 'max_prefill_tokens_per_chunk', 16384),
            }
            
            p = multiprocessing.Process(
                target=SGLangBackend._tp_worker_entry,
                args=(server_args_dict, rank, env, nccl_port)
            )
            p.start()
            self._tp_workers.append(p)
        
        # Load model on rank 0 (main process)
        from .sglang_batch_latency import load_model, _set_envs_and_config
        from sglang.srt.server_args import PortArgs
        
        env0 = os.environ.copy()
        env0["RANK"] = "0"
        env0["LOCAL_RANK"] = "0"
        env0["MASTER_PORT"] = str(nccl_port)
        for k, v in env0.items():
            os.environ[k] = v
            
        _set_envs_and_config(server_args_obj)
        port_args = PortArgs.init_new(server_args_obj)
        port_args.nccl_port = int(nccl_port)
        self._model_runner, self._tokenizer = load_model(server_args_obj, port_args, 0)

    @staticmethod
    def _tp_worker_entry(server_args_dict: Dict[str, Any], tp_rank: int, env: Dict[str, str], nccl_port: int):
        """Entry point for TP worker processes"""
        try:
            for k, v in env.items():
                os.environ[k] = v
            
            from .sglang_batch_latency import load_model, _set_envs_and_config
            from sglang.srt.server_args import PortArgs, ServerArgs
            
            # Recreate ServerArgs object from dict
            server_args_obj = ServerArgs(model_path=server_args_dict['model_path'])
            for k, v in server_args_dict.items():
                if hasattr(server_args_obj, k):
                    setattr(server_args_obj, k, v)
            
            _set_envs_and_config(server_args_obj)
            port_args = PortArgs.init_new(server_args_obj)
            port_args.nccl_port = int(nccl_port)
            _, _ = load_model(server_args_obj, port_args, tp_rank)
            
            # Keep worker alive
            while True:
                time.sleep(1)
        except Exception as e:
            print(f"TP worker {tp_rank} error: {e}")

    def destroy(self) -> None:
        # Cleanup TP workers
        for worker in self._tp_workers:
            try:
                worker.terminate()
                worker.join(timeout=5)
                if worker.is_alive():
                    worker.kill()
            except:
                pass
        self._tp_workers.clear()
        
        try:
            from sglang.srt.distributed.parallel_state import destroy_distributed_environment
            destroy_distributed_environment()
        except Exception:
            pass

class VLLMBackend(Backend):
    name = "vllm"

    def __init__(self) -> None:
        self._model_runner = None
        self._tokenizer = None
        self._loaded_sig: Optional[Tuple[str, str]] = None

    def init_env_and_nccl(self, server_args: Dict[str, Any]) -> int:
        # vLLM handles TP internally; no NCCL env wiring needed here.
        return 0

    def load_model_if_needed(
        self,
        model_path: str,
        server_args: Dict[str, Any],
        tp_rank: int,
        nccl_port: int,
    ) -> None:
        from .vllm_batch_latency import load_vllm_model, _set_vllm_envs_and_config
        from .bench_utils import hash_key

        sig = (model_path, hash_key(server_args))
        if self._loaded_sig == sig:
            return

        _set_vllm_envs_and_config(server_args)
        self._model_runner, self._tokenizer = load_vllm_model(model_path, server_args, tp_rank)
        self._loaded_sig = sig

    def prepare_inputs(self, batch_size: int, input_len: int) -> Any:
        from .vllm_batch_latency import prepare_vllm_synthetic_inputs
        return prepare_vllm_synthetic_inputs(batch_size, input_len)

    def run_once(
        self,
        run_name: str,
        reqs: Any,
        batch_size: int,
        input_len: int,
        output_len: int,
        chunk_prefill: bool = False,
        chunk_size: int = 512,
    ) -> Optional[Dict[str, Any]]:
        from .vllm_batch_latency import vllm_latency_test_run_once
        
        # Note: vllm doesn't support chunked prefill in this implementation
        # The parameters are accepted for interface compatibility but ignored
        return vllm_latency_test_run_once(
            run_name=run_name,
            model_runner=self._model_runner,
            rank_print=lambda *a, **k: None,
            reqs=reqs,
            batch_size=batch_size,
            input_len=input_len,
            output_len=output_len,
            device="cuda",
            log_decode_step=1000,
            profile=False,
            profile_filename_prefix="",
        )

    def destroy(self) -> None:
        pass
