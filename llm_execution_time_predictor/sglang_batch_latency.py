"""
Benchmark the latency of running a single static batch without a server.
# Usage (latency test)
## with dummy weights:
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --load-format dummy
## sweep through multiple data points and store (append) the results in a jsonl file:
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 12 14 --input-len 256 512 --output-len 32 256 --run-name test_run
## run with profiling:
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 12 14 --input-len 256 512 --profile
# Usage (correctness test):
python -m sglang.bench_one_batch --model-path TinyLlama/TinyLlama-1.1B-Chat-v0.4 --correct
"""

import argparse
import itertools
import json
import logging
import multiprocessing
import os
import time
from typing import Tuple, Any, List
from enum import Enum
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import destroy_distributed_environment
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    configure_logger,
    get_bool_env_var,
    kill_process_tree,
    set_gpu_proc_affinity,
)
from pathlib import Path
from llm_execution_time_predictor.profiling_utils import (
    synchronize,
    load_model,
    extend,
    decode,
    prepare_synthetic_inputs_for_latency_test,
)
from llm_execution_time_predictor.prefill_decode_isolated_profiler import (
    ForwardProfiler,
    ProfilingStrategy,
    PrefillStrategy,
    PrefillCacheStrategy,
    DecodeStrategy,
)
from llm_execution_time_predictor.args import BenchArgs


def prepare_inputs_for_correctness_test(
    bench_args: BenchArgs, tokenizer: Any
) -> List[Req]:
    prompts = [
        "The capital of France is",
        "The capital of the United Kindom is",
        "Today is a sunny day and I like",
    ]
    input_ids = [tokenizer.encode(p) for p in prompts]
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(prompts)):
        assert len(input_ids[i]) > bench_args.cut_len

        tmp_input_ids = input_ids[i][: bench_args.cut_len]
        req = Req(
            rid=i,
            origin_input_text=prompts[i],
            origin_input_ids=tmp_input_ids,
            sampling_params=sampling_params,
        )
        req.prefix_indices = []
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        reqs.append(req)

    return input_ids, reqs


def prepare_extend_inputs_for_correctness_test(
    bench_args, input_ids, reqs, model_runner
):
    for i in range(len(reqs)):
        req = reqs[i]
        req.fill_ids += input_ids[i][bench_args.cut_len :]
        req.prefix_indices = model_runner.req_to_token_pool.req_to_token[
            i, : bench_args.cut_len
        ]
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
    return reqs


def prefill_latency_test(
    server_args: Any, port_args: Any, bench_args: BenchArgs, tp_rank: int
) -> None:
    assert bench_args.run_prefill_profiling
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, tp_rank)
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    model_runner, tokenizer = load_model(server_args, port_args, tp_rank)
    profiling = ForwardProfiler(
        strategy=PrefillStrategy(
            batch_sizes=[1, 2, 4, 8, 16, 32, 48, 64, 72, 84, 128, 256],
            token_lengths=[
                1,
                2,
                4,
                8,
                16,
                32,
                64,
                128,
                256,
                512,
                1024,
                2048,
                4096,
                8192,
                10240,
                16384,
            ],
            skews=[0, 0.5, 1.0, 1.5],
            max_prefill_batch_size=bench_args.input_len[0],
        ),
        runner=model_runner,
        output_dir=Path(server_args.output_dir),
        prefix="prefill",
    )
    profiling.run()
    if server_args.tp_size > 1:
        destroy_distributed_environment()


def decode_latency_test(
    server_args: Any, port_args: Any, bench_args: BenchArgs, tp_rank: int
) -> None:
    assert bench_args.run_prefill_profiling
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, tp_rank)
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    model_runner, tokenizer = load_model(server_args, port_args, tp_rank)
    max_token_length = bench_args.input_len[0]
    output_dir = "profiling_results"
    profiling = ForwardProfiler(
        strategy=DecodeStrategy(
            batch_sizes=[1, 2, 4, 8, 16, 32, 48, 64, 72, 84, 128],
            token_lengths=[
                1,
                2,
                4,
                8,
                16,
                32,
                64,
                128,
                256,
                512,
                1024,
                2048,
                4096,
                8192,
                10240,
                16384,
                20480,
                32768,
                40960,
            ],
            skews=[0, 0.5, 1.0, 1.5, 2.0],
            max_tokens_limit=min(
                max_token_length, int(model_runner.max_total_num_tokens * 0.8)
            ),
        ),
        runner=model_runner,
        output_dir=Path(output_dir),
        prefix="decode",
    )
    profiling.run()
    if server_args.tp_size > 1:
        destroy_distributed_environment()


def prefill_latency_test_with_prefix_cache(
    server_args: Any, port_args: Any, bench_args: BenchArgs, tp_rank: int
) -> None:
    assert bench_args.run_prefill_profiling_with_prefix_cache
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, tp_rank)
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    model_runner, tokenizer = load_model(server_args, port_args, tp_rank)
    output_dir = "profiling_results"
    profiling = ForwardProfiler(
        strategy=PrefillCacheStrategy(
            batch_sizes=[1, 2, 4, 8, 16, 32, 48, 64, 72, 84, 128],
            token_lengths=[
                1,
                2,
                4,
                8,
                16,
                32,
                64,
                128,
                256,
                512,
                1024,
                2048,
                4096,
                8192,
                10240,
                16384,
            ],
            skews=[0.0, 0.5, 1.0, 1.5],
            cache_percents=[0, 0.25, 0.5, 0.75],
            chunked_flags=[False],
            max_prefill_batch_size=bench_args.input_len[0],
        ),
        runner=model_runner,
        output_dir=Path(output_dir),
        prefix="prefill_with_prefix_caching",
    )
    profiling.run()

    profiling = ForwardProfiler(
        strategy=PrefillCacheStrategy(
            batch_sizes=[1, 2, 4, 8, 16],
            token_lengths=[
                1,
                2,
                4,
                8,
                16,
                32,
                64,
                128,
                256,
                512,
                1024,
                2048,
                4096,
                8192,
                10240,
                16384,
            ],
            skews=[0.0, 0.5, 1.0, 1.5],
            cache_percents=[0.05, 0.5, 0.95],
            chunked_flags=[True],
            max_prefill_batch_size=bench_args.input_len[0],
        ),
        runner=model_runner,
        output_dir=Path(output_dir),
        prefix="prefill_profiling_chunked_cache_prefix_caching",
    )
    profiling.run()

    if server_args.tp_size > 1:
        destroy_distributed_environment()


def correctness_test(
    server_args: Any,
    port_args: Any,
    bench_args: BenchArgs,
    tp_rank: int,
) -> None:
    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, port_args, tp_rank)

    # Prepare inputs
    input_ids, reqs = prepare_inputs_for_correctness_test(bench_args, tokenizer)
    rank_print(f"\n{input_ids=}\n")

    if bench_args.cut_len > 0:
        # Prefill
        next_token_ids, next_token_logits, batch, prefill_iterations = extend(
            reqs, model_runner, bench_args.chunk_prefill, bench_args.chunk_size
        )
        rank_print(f"prefill logits (first half): {next_token_logits} \n")

    # Prepare extend inputs
    reqs = prepare_extend_inputs_for_correctness_test(
        bench_args, input_ids, reqs, model_runner
    )

    # Extend (prefill w/ KV cache)
    next_token_ids, next_token_logits, batch, prefill_iterations = extend(
        reqs, model_runner, bench_args.chunk_prefill, bench_args.chunk_size
    )
    rank_print(f"prefill logits (final): {next_token_logits} \n")

    # Decode
    output_ids = [input_ids[i] + [next_token_ids[i]] for i in range(len(input_ids))]
    for _ in range(bench_args.output_len[0] - 1):
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
        next_token_ids_list = next_token_ids.tolist()
        for i in range(len(reqs)):
            output_ids[i].append(next_token_ids_list[i])

    # Print output texts
    for i in range(len(reqs)):
        rank_print(f"========== Prompt {i} ==========")
        rank_print(tokenizer.decode(output_ids[i]), "\n")


def latency_test_run_once(
    run_name,
    model_runner,
    rank_print,
    reqs,
    batch_size,
    input_len,
    device,
    log_decode_step,
    profile,
    profile_filename_prefix,
    chunk_prefill=False,
    chunk_size=512,
    skew="none",
):
    model_config = model_runner.model_config

    max_batch_size = model_runner.max_total_num_tokens // (input_len + 1)
    if batch_size > max_batch_size:
        rank_print(
            f"skipping ({batch_size}, {input_len}, {1}) due to max batch size limit"
        )
        return

    # Clear the pools.
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    measurement_results = {
        "run_name": run_name,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": 1,
    }

    tot_latency = 0

    profiler = None
    if profile:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_stack=True,
        )
        profiler.start()

    # Prefill
    synchronize(device)
    tic = time.perf_counter()
    next_token_ids, _, batch, prefill_iterations = extend(
        reqs, model_runner, chunk_prefill, chunk_size
    )
    synchronize(device)
    prefill_latency = time.perf_counter() - tic
    tot_latency += prefill_latency
    throughput = input_len * batch_size / prefill_latency
    rank_print(
        f"Prefill. latency: {prefill_latency:6.5f} s, throughput: {throughput:9.2f} token/s, iterations: {prefill_iterations}"
    )
    measurement_results["prefill_latency"] = prefill_latency
    measurement_results["prefill_throughput"] = throughput
    measurement_results["prefill_iterations"] = prefill_iterations

    # Decode
    decode_latencies = []
    for i in range(1):
        synchronize(device)
        tic = time.perf_counter()
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
        synchronize(device)
        latency = time.perf_counter() - tic
        tot_latency += latency
        throughput = batch_size / latency
        decode_latencies.append(latency)
        if i < 5 or (log_decode_step > 0 and i % log_decode_step == 0):
            rank_print(
                f"Decode {i}. Batch size: {batch_size}, latency: {latency:6.5f} s, throughput: {throughput:9.2f} token/s"
            )
    output_len = 1

    if profile:
        profiler.stop()
        profile_filename = f"{profile_filename_prefix}_batch{batch_size}_input{input_len}_output{output_len}.trace.json.gz"
        parent_dir = os.path.dirname(os.path.abspath(profile_filename))
        os.makedirs(parent_dir, exist_ok=True)
        profiler.export_chrome_trace(profile_filename)
        rank_print(f"torch profiler chrome trace saved to {profile_filename}")
    # Record decode timing from 2nd output
    if output_len >= 1:
        med_decode_latency = np.median(decode_latencies)
        med_decode_throughput = batch_size / med_decode_latency
        rank_print(
            f"Decode.  median latency: {med_decode_latency:6.5f} s, median throughput: {med_decode_throughput:9.2f} token/s"
        )
        measurement_results["median_decode_latency"] = med_decode_latency
        measurement_results["median_decode_throughput"] = med_decode_throughput

    throughput = (input_len + output_len) * batch_size / tot_latency
    rank_print(
        f"Total. latency: {tot_latency:6.3f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["total_latency"] = tot_latency
    measurement_results["overall_throughput"] = throughput
    measurement_results["skew"] = skew
    return measurement_results


def latency_test(server_args, port_args, bench_args, tp_rank, disable_rank_print=True):
    # Set CPU affinity
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, tp_rank)

    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    if disable_rank_print:
        rank_print = lambda *args, **kwargs: None
    else:
        rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, port_args, tp_rank)

    # Prepare inputs for warm up
    reqs = prepare_synthetic_inputs_for_latency_test(
        bench_args.batch_size[0], bench_args.input_len[0], skew="none"
    )

    # Warm up
    rank_print("Warmup ...")
    latency_test_run_once(
        bench_args.run_name,
        model_runner,
        rank_print,
        reqs,
        bench_args.batch_size[0],
        bench_args.input_len[0],
        server_args.device,
        log_decode_step=0,
        profile=False,
        profile_filename_prefix="",  # not used
        chunk_prefill=bench_args.chunk_prefill,
        chunk_size=bench_args.chunk_size,
    )

    rank_print("Benchmark ...")

    # Run the sweep
    result_list = []
    for bs, il, ol in itertools.product(
        bench_args.batch_size, bench_args.input_len, bench_args.output_len
    ):
        reqs = prepare_synthetic_inputs_for_latency_test(bs, il, skew=bench_args.skew)
        ret = latency_test_run_once(
            bench_args.run_name,
            model_runner,
            rank_print,
            reqs,
            bs,
            il,
            server_args.device,
            bench_args.log_decode_step,
            bench_args.profile if tp_rank == 0 else None,
            bench_args.profile_filename_prefix,
            chunk_prefill=bench_args.chunk_prefill,
            chunk_size=bench_args.chunk_size,
            skew=bench_args.skew,
        )
        if ret is not None:
            result_list.append(ret)

    # Write results in jsonlines format on rank 0.
    if tp_rank == 0 and bench_args.result_filename:
        with open(bench_args.result_filename, "a") as fout:
            for result in result_list:
                fout.write(json.dumps(result) + "\n")

    if server_args.tp_size > 1:
        destroy_distributed_environment()


def main(server_args: Any, bench_args: BenchArgs) -> None:
    server_args.cuda_graph_max_bs = max(bench_args.batch_size)

    _set_envs_and_config(server_args)

    if server_args.model_path:
        if bench_args.correctness_test:
            work_func = correctness_test
        elif bench_args.run_prefill_profiling:
            work_func = prefill_latency_test
        elif bench_args.run_decode_profiling:
            work_func = decode_latency_test
        elif bench_args.run_prefill_profiling_with_prefix_cache:
            work_func = prefill_latency_test_with_prefix_cache
        else:
            work_func = latency_test

    else:
        raise ValueError(
            "Provide --model-path for running the tests or "
            "provide --result-filename for plotting the results"
        )

    port_args = PortArgs.init_new(server_args)

    if server_args.tp_size == 1:
        work_func(server_args, port_args, bench_args, 0)
    else:
        workers = []
        for tp_rank in range(server_args.tp_size):
            proc = multiprocessing.Process(
                target=work_func,
                args=(
                    server_args,
                    port_args,
                    bench_args,
                    tp_rank,
                ),
            )
            proc.start()
            workers.append(proc)

        for proc in workers:
            proc.join()


def parse_cli_args_main() -> Tuple[Any, BenchArgs]:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        main(server_args, bench_args)
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    parse_cli_args_main()
