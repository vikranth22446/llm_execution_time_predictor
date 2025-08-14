import fcntl
import json
import os
import threading
import time
from contextlib import suppress
from queue import Empty, Queue
from typing import Any, Dict, Optional

import torch

if os.environ.get("SGLANG_ENABLE_PROFILING") == "1":
    write_queue = Queue()
    pid = os.getpid()

    def background_writer() -> None:
        output_file = os.environ.get(
            "SGLANG_PROFILE_OUTPUT", "model_runner_profile.jsonl"
        )

        flush_interval = 2.0  # seconds
        last_flush = time.time()
        buffer = []

        while True:
            now = time.time()
            try:
                item = write_queue.get(timeout=1)
                if item is None:
                    break
                buffer.append(item)
            except Empty:
                pass

            if buffer and (len(buffer) >= 20 or now - last_flush >= flush_interval):
                try:
                    with open(output_file, "a") as f:
                        fcntl.flock(f, fcntl.LOCK_EX)
                        try:
                            for record in buffer:
                                f.write(json.dumps(record) + "\n")
                            f.flush()  # make sure data is written
                        finally:
                            fcntl.flock(f, fcntl.LOCK_UN)
                    buffer.clear()
                    last_flush = now
                except Exception as e:
                    # Could log this
                    pass

        # Final flush
        if buffer:
            with suppress(Exception):
                with open(output_file, "a") as f:
                    for record in buffer:
                        f.write(json.dumps(record) + "\n")

    threading.Thread(target=background_writer, daemon=True).start()

    try:
        from sglang.srt.model_executor.model_runner import ModelRunner

        # For Refernece with Monkey patching: https://github.com/sgl-project/sglang/blob/c1d2061f97ae2facb8edbc188da5b719aaac2f09/python/sglang/srt/model_executor/model_runner.py#L1622
        # Change depending on SGLang version/refactoring
        # ForwardMode/Batch Properties Reference: https://github.com/sgl-project/sglang/blob/c1d2061f97ae2facb8edbc188da5b719aaac2f09/python/sglang/srt/model_executor/forward_batch_info.py#L68
        if not hasattr(ModelRunner.forward, "_sglang_profiled"):
            original = ModelRunner.forward

            def profiled_forward(self, *args: Any, **kwargs: Any) -> Any:
                torch.cuda.synchronize()
                start = time.perf_counter()
                batch = args[0] if args else None
                result = original(self, *args, **kwargs)
                # Synchronize
                torch.cuda.synchronize()
                duration = time.perf_counter() - start

                tokens = getattr(batch, "total_num_tokens", 0) or 0
                forward_mode = getattr(batch, "forward_mode", 1)

                if hasattr(forward_mode, "is_extend"):
                    forward_mode_str = (
                        "prefill" if forward_mode.is_extend() else "decode"
                    )

                total_seq_lens = getattr(batch, "seq_lens", [])
                if hasattr(total_seq_lens, "cpu"):
                    total_seq_lens = total_seq_lens.cpu().tolist()

                extend_seq_lens = getattr(batch, "extend_seq_lens_cpu", [])
                prefix_lens = getattr(batch, "extend_prefix_lens_cpu", [])

                if forward_mode_str == "decode":
                    extend_seq_lens = [1] * len(total_seq_lens)
                    prefix_lens = (
                        [seq_len - 1 for seq_len in total_seq_lens]
                        if total_seq_lens
                        else []
                    )

                write_queue.put(
                    {
                        "timestamp": start,
                        "process_id": os.getpid(),
                        "batch_size": getattr(batch, "batch_size", 0),
                        "combined_seq_lens": total_seq_lens,  # Total sequence lengths (cached + new)
                        "cached_prefix_lens": prefix_lens,  # Length of cached prefix portion
                        "new_extend_lens": extend_seq_lens,  # Length of new tokens being added
                        "total_token_length": sum(total_seq_lens)
                        if total_seq_lens
                        else 0,
                        "forward_mode": forward_mode_str,
                        "latency": duration,
                    }
                )
                return result

            profiled_forward._sglang_profiled = True
            ModelRunner.forward = profiled_forward
    except ImportError:
        pass
