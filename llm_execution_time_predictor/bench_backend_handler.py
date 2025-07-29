from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

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

    @property
    def supports_external_tp(self) -> bool:
        return True

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

        # hash server_args for caching
        import json, hashlib
        sig = (model_path, hashlib.md5(json.dumps(server_args, sort_keys=True, default=str).encode()).hexdigest())
        if self._loaded_sig == sig:
            return

        server_args_obj = ServerArgs(model_path=model_path)
        for k, v in server_args.items():
            if hasattr(server_args_obj, k):
                setattr(server_args_obj, k, v)
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
        )

    def destroy(self) -> None:
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

        import json, hashlib
        sig = (model_path, hashlib.md5(json.dumps(server_args, sort_keys=True, default=str).encode()).hexdigest())
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
    ) -> Optional[Dict[str, Any]]:
        from .vllm_batch_latency import vllm_latency_test_run_once

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
