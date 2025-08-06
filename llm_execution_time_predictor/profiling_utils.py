import numpy as np
import json
import torch
from typing import List, Optional, Tuple, Any, Union
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    require_mlp_sync,
    require_mlp_tp_gather,
    suppress_other_loggers,
)
from llm_execution_time_predictor.args import BenchArgs
from sglang.srt.configs.model_config import ModelConfig
import time
from sglang.srt.hf_transformers_utils import get_tokenizer
import torch.distributed as dist


@torch.no_grad
def extend(
    reqs: List[Req], model_runner: ModelRunner
) -> Tuple[List[int], torch.Tensor, ScheduleBatch, int]:
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=None,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        enable_custom_logit_processor=False,
    )
    batch.prepare_for_extend()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output, _ = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return (
        next_token_ids,
        logits_output.next_token_logits,
        batch,
        1,
    )  # 1 prefill iteration


@torch.no_grad
def decode(
    input_token_ids: torch.Tensor, batch: ScheduleBatch, model_runner: ModelRunner
) -> Tuple[List[int], torch.Tensor]:
    batch.output_ids = input_token_ids
    batch.prepare_for_decode()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output, _ = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits


def _maybe_prepare_mlp_sync_batch(
    batch: ScheduleBatch, model_runner: ModelRunner
) -> None:
    if require_mlp_sync(model_runner.server_args):
        Scheduler.prepare_mlp_sync_batch_raw(
            batch,
            dp_size=model_runner.server_args.dp_size,
            attn_tp_size=1,
            tp_group=model_runner.tp_group,
            get_idle_batch=None,
            disable_cuda_graph=model_runner.server_args.disable_cuda_graph,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            speculative_num_draft_tokens=None,
            require_mlp_tp_gather=require_mlp_tp_gather(model_runner.server_args),
            disable_overlap_schedule=model_runner.server_args.disable_overlap_schedule,
        )


def prepare_synthetic_inputs_for_latency_test(
    batch_size: int, input_len: int, input_len_list: Optional[List[int]] = None
) -> List[Req]:
    """
    Prepare synthetic inputs for latency testing.

    Args:
        batch_size: Number of requests in the batch
        input_len: Default input length (used when input_len_list is None)
        input_len_list: Optional list of input lengths for each request in the batch.
                       If provided, batch_size must match len(input_len_list).
        skew: Distribution skew for input lengths ("none", "medium", "heavy")
    """
    if input_len_list is not None:
        input_ids = [
            np.random.randint(0, 10000, size=(item,), dtype=np.int32)
            for item in input_len_list
        ]
    else:
        input_ids = np.random.randint(0, 10000, (batch_size, input_len), dtype=np.int32)

    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )
    reqs = []
    for i in range(batch_size):
        current_input_ids = list(input_ids[i])
        if len(current_input_ids) == 0:
            continue
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=current_input_ids,
            sampling_params=sampling_params,
        )
        req.prefix_indices = []
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        reqs.append(req)
    return reqs


def warmup_model(model_runner: ModelRunner) -> None:
    np.random.seed(42)
    warmup_reqs = prepare_synthetic_inputs_for_latency_test(
        batch_size=1, input_len=1024, input_len_list=[1024]
    )
    synchronize(model_runner.device)
    (next_token_ids, _, batch, _) = extend(warmup_reqs, model_runner)
    synchronize(model_runner.device)

    decode(
        torch.tensor(next_token_ids, device=model_runner.device),
        batch,
        model_runner,
    )
    synchronize(model_runner.device)


def run_decoding_config(
    model_runner: ModelRunner, next_input_ids: torch.Tensor, batch: ScheduleBatch
) -> Tuple[float, float]:
    """
    Run the decoding with a specific skewed batch configuration.
    This function is used to profile the decoding performance with different skew configurations.
    """
    synchronize(model_runner.device)
    start_time = time.perf_counter()
    next_token_ids, _ = decode(next_input_ids, batch, model_runner)
    synchronize(model_runner.device)
    end_time = time.perf_counter()
    latency = end_time - start_time
    total_tokens = len(batch.reqs)
    throughput = total_tokens / latency
    return latency, throughput


def synchronize(device: Union[str, torch.device]) -> None:
    torch.get_device_module(device).synchronize()


def run_prefill_config(
    model_runner: ModelRunner,
    skewed_batch_lens: List[int],
    prefix_cached_lengths: Optional[List[int]] = None,
    clear_cache: bool = True,
    chunk_prefill_size: int = 16384,
    reqs: Optional[List[Req]] = None,
) -> Tuple[float, float, Tuple[List[int], ScheduleBatch]]:
    """Run prefill with optional prefix cache. Returns (latency, throughput, (tokens, batch)).
    clear_cache=False preserves KV cache between calls.
    """
    if clear_cache:
        model_runner.req_to_token_pool.clear()
        model_runner.token_to_kv_pool_allocator.clear()
    if reqs is None:
        reqs = prepare_synthetic_inputs_for_latency_test(
            len(skewed_batch_lens), max(skewed_batch_lens), skewed_batch_lens
        )
    if prefix_cached_lengths is not None:
        reqs_non_zero = [
            req for req, pre_len in zip(reqs, prefix_cached_lengths) if pre_len > 0
        ]

        if len(reqs_non_zero) != 0:
            for i, req in enumerate(reqs_non_zero):
                req.extend_input_len = prefix_cached_lengths[i]
                req.fill_ids = req.fill_ids[: len(req.prefix_indices) + req.extend_input_len]
            run_prefill_in_chunks_to_load_cache(
                model_runner, prefix_cached_lengths[: len(reqs_non_zero)], reqs_non_zero, chunk_size=chunk_prefill_size
            )
    synchronize(model_runner.device)
    start_time = time.perf_counter()
    next_token_ids, _, batch, _ = extend(reqs, model_runner)
    synchronize(model_runner.device)

    end_time = time.perf_counter()
    latency = end_time - start_time
    if not clear_cache:
        for req in reqs:
            if len(req.fill_ids) != 0:
                # Filled length 
                cached_length = len(req.fill_ids)
                req.fill_ids = req.origin_input_ids + req.output_ids  
                req.prefix_indices = model_runner.req_to_token_pool.req_to_token[  
                    req.req_pool_idx, : len(req.prefix_indices) + cached_length  
                ]  
                req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)  
                req.logprob_start_len = len(req.origin_input_ids) - 1

    total_tokens = sum(skewed_batch_lens)
    throughput = total_tokens / latency
    return latency, throughput, (next_token_ids, batch)


def run_prefill_in_chunks_to_load_cache(
    model_runner: ModelRunner,
    skewed_batch_lens: List[int],
    reqs: List[Req],
    chunk_size: int = 16384,
) -> Tuple[ScheduleBatch, List[int]]:
    """Load KV cache by chunking sequences up to chunk_size tokens.
    Returns (merged_batch, all_next_tokens).
    """
    start_index = 0
    all_batches = []
    all_next_token_ids = []

    while start_index < len(skewed_batch_lens):
        current_sum = 0
        end_index = start_index

        while (
            end_index < len(skewed_batch_lens)
            and current_sum + skewed_batch_lens[end_index] <= chunk_size
        ):
            current_sum += skewed_batch_lens[end_index]
            end_index += 1

        if end_index == start_index:
            end_index += 1

        current_chunk = skewed_batch_lens[start_index:end_index]
        current_reqs = reqs[start_index:end_index]

        latency, throughput, (next_token_ids, batch) = run_prefill_config(
            model_runner, current_chunk, clear_cache=False, reqs=current_reqs
        )
        all_batches.append(batch)
        all_next_token_ids.extend(next_token_ids)

        start_index = end_index  # advance to next chunk

    batch0: ScheduleBatch = all_batches[0]
    for batch in all_batches[1:]:
        batch0.merge_batch(batch)

    return batch0, all_next_token_ids


def generate_distribution_of_skewed_batch(
    total_elements: int, batch_size: int, skew: float
) -> List[int]:
    """Generate Zipf-like distribution. skew=0 is uniform, higher=more skewed.
    Example: (100, 4, 1.0) -> [40, 20, 20, 20]
    """
    if batch_size == 1:
        return [total_elements]

    if skew == 0:
        counts = np.ones(batch_size)
    else:
        ranks = np.arange(1, batch_size + 1)
        counts = 1 / ranks**skew

    counts *= total_elements / np.sum(counts)
    counts_rounded = np.floor(counts).astype(int)
    remainder = total_elements - np.sum(counts_rounded)
    if remainder > 0:
        # Adds the remainder to the top `remainder` elements
        frac_parts = counts - counts_rounded
        top_indices = np.argpartition(-frac_parts, remainder)[:remainder]
        counts_rounded[top_indices] += 1
    counts_rounded = [int(x) for x in counts_rounded]
    return counts_rounded


def generate_distribution_skewed_batch_with_prefix_cache(
    total_elements: int,
    batch_size: int,
    skew: float,
    prefix_cache_percent: float,
    chunked_prefill_distribution: bool = False,
) -> Tuple[List[int], List[int]]:
    """
    Returns (sequence_lengths, prefix_cache_lengths)
    With chunked prefill, a very common case is prefix cache ends up being
    [x, 0, 0, 0] where x is borrowed from the previous iteration.

    The remaining entries model extend kernels with varying amounts of prefix used for general training.
    """
    skewed_lengths = generate_distribution_of_skewed_batch(
        total_elements, batch_size, skew
    )
    if chunked_prefill_distribution:
        return skewed_lengths, [int(skewed_lengths[0] * prefix_cache_percent)] + [0] * (
            batch_size - 1
        )
    total_cached_tokens = int(total_elements * prefix_cache_percent)
    prefix_cache_skewed_lengths = generate_distribution_of_skewed_batch(
        total_cached_tokens, batch_size, skew
    )
    assert len(skewed_lengths) == len(prefix_cache_skewed_lengths), (
        "Both distributions must have the same batch size."
    )
    remaining_prefix_lengths = [
        min(skewed_lengths[i], prefix_cache_skewed_lengths[i])
        for i in range(len(skewed_lengths))
    ]
    remaining = total_cached_tokens - sum(remaining_prefix_lengths)
    if remaining > 0:
        capacities = [
            (skewed_lengths[i] - remaining_prefix_lengths[i], i)
            for i in range(len(skewed_lengths))
        ]
        capacities.sort(reverse=True, key=lambda x: x[0])
        for cap, idx in capacities:
            if remaining <= 0:
                break
            if cap <= 0:
                continue
            remaining_prefix_lengths[idx] += 1
            remaining -= 1
    return skewed_lengths, remaining_prefix_lengths


def get_rank_print(model_runner: ModelRunner):
    return print if model_runner.tp_rank == 0 else lambda *args, **kwargs: None


def write_results_to_file(results: List[Any], filename: str) -> None:
    with open(filename, "w") as fout:
        for result in results:
            fout.write(json.dumps(result) + "\n")


def filter_token_lengths(lengths: List[int], max_length: int) -> List[int]:
    return [x for x in lengths if x <= max_length]


def load_model(server_args, port_args, tp_rank: int) -> Tuple[ModelRunner, Any]:
    suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    model_config = ModelConfig.from_server_args(server_args)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=tp_rank,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        pp_rank=0,
        pp_size=server_args.pp_size,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    rank_print(f"max_total_num_tokens={model_runner.max_total_num_tokens}")
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    if server_args.tp_size > 1:
        dist.barrier()
    return model_runner, tokenizer

def create_profiling_result_dic(
    batch_size: int,
    total_token_length: int,
    skew: float,
    combined_seq_lens: list,
    cached_prefix_lens: list,
    new_extend_lens: list,
    latency: float,
    throughput: float,
    forward_mode: str,
    **optional_fields
) -> dict:
    result = {
        "batch_size": batch_size,
        "total_token_length": total_token_length,
        "skew": skew,
        "combined_seq_lens": combined_seq_lens,
        "cached_prefix_lens": cached_prefix_lens,
        "new_extend_lens": new_extend_lens,
        "total_extend_len": sum(new_extend_lens),
        "latency": latency,
        "throughput": throughput,
        "forward_mode": forward_mode,
    }
    result.update(optional_fields)
    return result