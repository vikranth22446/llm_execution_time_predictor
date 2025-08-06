import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterator, Iterable, Optional
from tqdm import tqdm
from llm_execution_time_predictor.profiling_utils import (
    write_results_to_file,
    generate_distribution_of_skewed_batch,
    generate_distribution_skewed_batch_with_prefix_cache,
    run_prefill_config,
    warmup_model,
    prepare_synthetic_inputs_for_latency_test,
    run_prefill_in_chunks_to_load_cache,
    run_decoding_config,
    filter_token_lengths,
)

logger = logging.getLogger(__name__)


def _config_product(config_cls: type, *iterables: Iterable[Any]) -> Iterator[Any]:
    return (config_cls(*args) for args in product(*iterables))


class ProfilingStrategy(ABC):
    @abstractmethod
    def generate_configs(self) -> Iterator[Any]:
        raise NotImplementedError

    @abstractmethod
    def run_one(self, runner: Any, cfg: Any) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def output_filename(self, prefix: str, tp_rank: int) -> str:
        raise NotImplementedError


class ForwardProfiler:
    def __init__(
        self, strategy: ProfilingStrategy, runner: Any, output_dir: Path, prefix: str
    ):
        self.strategy = strategy
        self.runner = runner
        self.output_dir = output_dir
        self.prefix = prefix
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        logger.info("Warmup for profiling...")
        warmup_model(self.runner)

        configs = list(self.strategy.generate_configs())
        logger.info("Profiling %d configurations.", len(configs))

        results: list[Dict[str, Any]] = []
        for cfg in tqdm(configs, desc=f"{self.prefix} Profiling", unit="run"):
            results.append(self.strategy.run_one(self.runner, cfg))

        filename = self.strategy.output_filename(self.prefix, self.runner.tp_rank)
        out_path = self.output_dir / filename
        write_results_to_file(results, str(out_path))
        logger.info("Results written to %s", out_path)


# --- Config dataclasses ---
@dataclass(frozen=True)
class PrefillConfig:
    batch_size: int
    token_length: int
    skew: float


@dataclass(frozen=True)
class PrefillCacheConfig:
    batch_size: int
    token_length: int
    skew: float
    cache_percent: float
    chunked: bool


@dataclass(frozen=True)
class DecodeConfig:
    batch_size: int
    token_length: int
    skew: float


class PrefillStrategy(ProfilingStrategy):
    """
    Regular prefill profiling without prefix cache.
    """

    def __init__(
        self,
        batch_sizes: Iterable[int],
        token_lengths: Iterable[int],
        skews: Iterable[float],
        max_prefill_batch_size: Optional[int] = None,
    ):
        self.batch_sizes = list(batch_sizes)
        self.token_lengths = list(token_lengths)
        self.skews = list(skews)
        self.token_lengths = filter_token_lengths(
            self.token_lengths,
            max_prefill_batch_size if max_prefill_batch_size is not None else int(1e9),
        )

    def generate_configs(self) -> Iterator[PrefillConfig]:
        return _config_product(
            PrefillConfig, self.batch_sizes, self.token_lengths, self.skews
        )

    def run_one(self, runner: Any, cfg: PrefillConfig) -> Dict[str, Any]:
        skewed = generate_distribution_of_skewed_batch(
            cfg.token_length, cfg.batch_size, cfg.skew
        )
        latency, throughput, _ = run_prefill_config(runner, skewed)
        return {
            "batch_size": cfg.batch_size,
            "total_token_length": cfg.token_length,
            "skew": cfg.skew,
            "latency": latency,
            "throughput": throughput,
            "forward_mode": "prefill",
        }

    def output_filename(self, prefix: str, tp_rank: int) -> str:
        return f"prefill_{prefix}_profiling_tp{tp_rank}.jsonl"


class PrefillCacheStrategy(ProfilingStrategy):
    """
    Prefill profiling with prefix cache scenarios.
    """

    def __init__(
        self,
        batch_sizes: Iterable[int],
        token_lengths: Iterable[int],
        skews: Iterable[float],
        cache_percents: Iterable[float],
        chunked_flags: Iterable[bool],
        max_prefill_batch_size: Optional[int] = None,
    ):
        self.batch_sizes = list(batch_sizes)
        self.token_lengths = list(token_lengths)
        self.skews = list(skews)
        self.cache_percents = list(cache_percents)
        self.chunked_flags = list(chunked_flags)
        self.token_lengths = filter_token_lengths(
            self.token_lengths,
            max_prefill_batch_size if max_prefill_batch_size is not None else int(1e9),
        )

    def generate_configs(self) -> Iterator[PrefillCacheConfig]:
        return _config_product(
            PrefillCacheConfig,
            self.batch_sizes,
            self.token_lengths,
            self.skews,
            self.cache_percents,
            self.chunked_flags,
        )

    def run_one(self, runner: Any, cfg: PrefillCacheConfig) -> Dict[str, Any]:
        skewed, prefix_cache = generate_distribution_skewed_batch_with_prefix_cache(
            cfg.token_length,
            cfg.batch_size,
            cfg.skew,
            prefix_cache_percent=cfg.cache_percent,
            chunked_prefill_distribution=cfg.chunked,
        )
        print("Running config with prefix cache:", cfg)
        latency, throughput, _ = run_prefill_config(
            runner, skewed, prefix_cached_lengths=prefix_cache, chunk_prefill_size=16384
        )
        return {
            "batch_size": cfg.batch_size,
            "total_token_length": cfg.token_length,
            "skew": cfg.skew,
            "prefix_cache": prefix_cache,
            "cache_percent": cfg.cache_percent,
            "chunked": cfg.chunked,
            "latency": latency,
            "throughput": throughput,
            "forward_mode": "prefill_cache",
        }

    def output_filename(self, prefix: str, tp_rank: int) -> str:
        return f"{prefix}_prefill_cache_profiling_tp{tp_rank}.jsonl"


class DecodeStrategy(ProfilingStrategy):
    """
    Decode profiling after loading KV cache.
    """

    def __init__(
        self,
        batch_sizes: Iterable[int],
        token_lengths: Iterable[int],
        skews: Iterable[float],
        max_tokens_limit: Optional[int] = None,
    ):
        self.batch_sizes = list(batch_sizes)
        self.token_lengths = list(token_lengths)
        self.skews = list(skews)
        self.token_lengths = filter_token_lengths(token_lengths, max_tokens_limit)

    def generate_configs(self) -> Iterator[DecodeConfig]:
        return _config_product(
            DecodeConfig, self.batch_sizes, self.token_lengths, self.skews
        )

    def _prepare_cache(self, runner: Any, cfg: DecodeConfig) -> tuple[Any, Any]:
        skewed, prefix_cache = generate_distribution_skewed_batch_with_prefix_cache(
            cfg.token_length,
            cfg.batch_size,
            cfg.skew,
            prefix_cache_percent=0.0,
            chunked_prefill_prefix_cache=False,
        )
        reqs = prepare_synthetic_inputs_for_latency_test(
            batch_size=cfg.batch_size, input_len=cfg.token_length, input_len_list=skewed
        )
        return run_prefill_in_chunks_to_load_cache(runner, skewed, reqs)

    def run_one(self, runner: Any, cfg: DecodeConfig) -> Dict[str, Any]:
        batch, next_ids = self._prepare_cache(runner, cfg)
        latency, throughput = run_decoding_config(runner, next_ids, batch)
        return {
            "batch_size": cfg.batch_size,
            "total_token_length": cfg.token_length,
            "skew": cfg.skew,
            "latency": latency,
            "throughput": throughput,
            "forward_mode": "decode",
        }

    def output_filename(self, prefix: str, tp_rank: int) -> str:
        return f"{prefix}_decode_profiling_tp{tp_rank}.jsonl"
