import torch
import os
import time
import json
import threading
from queue import Queue, Empty
from contextlib import suppress
import fcntl

if os.environ.get('SGLANG_ENABLE_PROFILING') == '1':
    write_queue = Queue()
    pid = os.getpid()
    def background_writer():
        output_file = os.environ.get('SGLANG_PROFILE_OUTPUT', 'model_runner_profile.jsonl')
        
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
                    with open(output_file, 'a') as f:
                        fcntl.flock(f, fcntl.LOCK_EX)
                        try:
                            for record in buffer:
                                f.write(json.dumps(record) + '\n')
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
                with open(output_file, 'a') as f:
                    for record in buffer:
                        f.write(json.dumps(record) + '\n')

    threading.Thread(target=background_writer, daemon=True).start()

    try:
        from sglang.srt.model_executor.model_runner import ModelRunner
        # For Refernece with Monkey patching: https://github.com/sgl-project/sglang/blob/c1d2061f97ae2facb8edbc188da5b719aaac2f09/python/sglang/srt/model_executor/model_runner.py#L1622
        # Change depending on SGLang version/refactoring
        # ForwardMode/Batch Properties Reference: https://github.com/sgl-project/sglang/blob/c1d2061f97ae2facb8edbc188da5b719aaac2f09/python/sglang/srt/model_executor/forward_batch_info.py#L68
        if not hasattr(ModelRunner.forward, '_sglang_profiled'):
            original = ModelRunner.forward

            def profiled_forward(self, *args, **kwargs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                batch = args[0] if args else None
                result = original(self, *args, **kwargs)
                # Synchronize
                torch.cuda.synchronize()
                duration = time.perf_counter() - start

                tokens = getattr(batch, 'total_num_tokens', 0) or 0
                forward_mode = getattr(batch, 'forward_mode', 1)
                # 
                if hasattr(forward_mode, 'is_extend'):  
                    forward_mode_str = 'prefill' if forward_mode.is_extend() else 'decode'  

                seq_lens = getattr(batch, 'extend_seq_lens_cpu', []) if forward_mode_str == "prefill" else getattr(batch, 'seq_lens_cpu', [])
                if seq_lens is None or len(seq_lens) == 0:  
                    seq_lens = getattr(batch, 'seq_lens', [])  
                    if hasattr(seq_lens, 'cpu'):  
                        seq_lens = seq_lens.cpu().tolist()
                    
                write_queue.put({
                    "timestamp": start,
                    "process_id": os.getpid(),
                    "batch_size": getattr(batch, 'batch_size', 0),
                    "batch_lens": seq_lens,
                    "total_token_length": sum(seq_lens) if seq_lens else 0,
                    "forward_mode": forward_mode_str,
                    "latency": duration,
                })
                return result

            profiled_forward._sglang_profiled = True
            ModelRunner.forward = profiled_forward
    except ImportError:
        pass
