"""
VLLM-specific benchmarking module that mirrors the sglang interface.

This module provides VLLM benchmarking functions compatible with the batch_benchmark_runner.
Key differences from sglang:
- No multiprocessing needed (VLLM handles tensor parallelism internally)
- Cache reset only at end of each test run
- Simplified model loading with tensor_parallel_size parameter
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional

def _set_vllm_envs_and_config(server_args):
    """Configure VLLM environment - much simpler than sglang."""
    # VLLM handles most configuration internally
    pass

def load_vllm_model(model_path: str, server_args: Dict[str, Any], tp_rank: int):
    """
    Load VLLM model with specified configuration.
    
    Args:
        model_path: Path to the model
        server_args: Server configuration arguments
        tp_rank: Tensor parallel rank (always 0 for VLLM since it handles TP internally)
    
    Returns:
        Tuple of (llm, tokenizer)
    """
    from vllm import LLM
    from transformers import AutoTokenizer
    
    # Extract VLLM-specific arguments
    tensor_parallel_size = server_args.get("tp_size", 1)
    load_format = server_args.get("load_format", "auto")
    
    # Map sglang load_format to VLLM equivalent
    if load_format == "dummy":
        load_format = "dummy"
    else:
        load_format = "auto"
    
    # Load model with tensor parallelism
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        load_format=load_format, 
        max_num_batched_tokens=65536,
        max_num_seqs=64
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return llm, tokenizer

def prepare_vllm_synthetic_inputs(batch_size: int, input_len: int, input_ids_list: Optional[List] = None):
    """
    Prepare synthetic inputs for VLLM latency testing.
    
    Args:
        batch_size: Number of requests in the batch
        input_len: Default input length (used when input_ids_list is None)
        input_ids_list: Optional list of input_ids arrays for each request in the batch
    
    Returns:
        List of input configurations for VLLM
    """
    if input_ids_list is not None:
        if len(input_ids_list) != batch_size:
            raise ValueError(f"Length of input_ids_list ({len(input_ids_list)}) must match batch_size ({batch_size})")
        input_ids = input_ids_list
    else:
        # Generate random input_ids with same length
        input_ids = np.random.randint(0, 10000, (batch_size, input_len), dtype=np.int32)
    
    # Convert to list of lists for VLLM
    reqs = []
    for i in range(batch_size):
        current_input_ids = list(input_ids[i])
        reqs.append({
            'input_ids': current_input_ids,
            'rid': i
        })
    
    return reqs

def vllm_latency_test_run_once(
    run_name: str,
    model_runner,  # This will be the VLLM LLM instance
    rank_print,
    reqs: List[Dict],
    batch_size: int,
    input_len: int,
    output_len: int,
    device: str,
    log_decode_step: int,
    profile: bool,
    profile_filename_prefix: str,
) -> Optional[Dict[str, Any]]:
    """
    Run a single VLLM benchmark test with proper prefill/decode timing separation.
    
    Args:
        run_name: Name of the benchmark run
        model_runner: VLLM LLM instance 
        rank_print: Print function for logging
        reqs: List of request dictionaries with 'input_ids'
        batch_size: Batch size
        input_len: Input length
        output_len: Output length
        device: Device string (for compatibility)
        log_decode_step: Decode logging step
        profile: Whether to profile
        profile_filename_prefix: Profile filename prefix
    
    Returns:
        Dictionary with benchmark results or None if skipped
    """
    from vllm import SamplingParams
    
    llm = model_runner  # VLLM LLM instance
    
    # TODO Fix timings for vLLM to match sglang
    prompt_token_ids = []
    for req in reqs:
        token_ids = req['input_ids']
        # Ensure token_ids is a list of integers
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        elif not isinstance(token_ids, list):
            token_ids = list(token_ids)
        token_ids = [int(x) for x in token_ids]
        prompt_token_ids.append(token_ids)
    
    predicted_kv_usage_in_gb = batch_size * (input_len + output_len) * 16 * 32 * 4096 * 2 / (1024**3)  # Rough estimate
    
    rank_print(
        f"Predicted peak KV cache usage: {predicted_kv_usage_in_gb:.3f} GB (VLLM estimate)"
    )
    
    measurement_results = {
        "run_name": run_name,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
    }
    
    try:
        # Warmup run
        sampling_params_warmup = SamplingParams(max_tokens=1, temperature=0.0)
        _ = llm.generate(prompt_token_ids=[prompt_token_ids[0]], sampling_params=sampling_params_warmup)
        
        sampling_params_prefill = SamplingParams(max_tokens=1, temperature=0.0)
        start_time = time.time()
        prefill_outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params_prefill)
        prefill_latency = time.time() - start_time
        
        prefill_throughput = input_len * batch_size / prefill_latency
        rank_print(
            f"Prefill. latency: {prefill_latency:6.5f} s, throughput: {prefill_throughput:9.2f} token/s"
        )
        measurement_results["prefill_latency"] = prefill_latency
        measurement_results["prefill_throughput"] = prefill_throughput
        
        # Full generation for decode timing (using same cache state)
        # This includes prefill + decode, so we'll subtract prefill time
        sampling_params_full = SamplingParams(max_tokens=output_len, temperature=0.0)
        start_time = time.time()
        full_outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params_full)
        total_latency = time.time() - start_time
        
        # Calculate decode latency
        # Note: This is approximate since the second run may have different prefill timing
        # But it's the best we can do with VLLM's API while maintaining cache consistency
        decode_latency = total_latency - prefill_latency
        # Calculate per-token decode latency (excluding the first token which was prefilled)
        if output_len > 1:
            median_decode_latency = decode_latency / (output_len - 1)
            median_decode_throughput = batch_size / median_decode_latency if median_decode_latency > 0 else 0
            rank_print(
                f"Decode.  median latency: {median_decode_latency:6.5f} s, median throughput: {median_decode_throughput:9.2f} token/s"
            )
            measurement_results["median_decode_latency"] = median_decode_latency
            measurement_results["median_decode_throughput"] = median_decode_throughput
        else:
            measurement_results["median_decode_latency"] = 0
            measurement_results["median_decode_throughput"] = 0
        
        # Overall metrics
        overall_throughput = (input_len + output_len) * batch_size / total_latency
        rank_print(
            f"Total. latency: {total_latency:6.3f} s, throughput: {overall_throughput:9.2f} token/s"
        )
        measurement_results["total_latency"] = total_latency
        measurement_results["overall_throughput"] = overall_throughput
        measurement_results["predicted_kv_usage_in_gb"] = predicted_kv_usage_in_gb
        
        # Reset cache at END of test run
        llm.reset_prefix_cache()
        
        return measurement_results
        
    except Exception as e:
        rank_print(f"Error in VLLM benchmark: {e}")
        # Reset cache even on error
        try:
            llm.reset_prefix_cache()
        except:
            pass
        raise e