import os
import time
import json
import threading
from queue import Queue, Empty
from contextlib import suppress

if os.environ.get('SGLANG_ENABLE_PROFILING') == '1':
    write_queue = Queue()
    
    def background_writer():
        base_file = os.environ.get('SGLANG_PROFILE_OUTPUT', 'model_runner_profile.jsonl')
        output_file = f"{os.path.splitext(base_file)[0]}_pid_{os.getpid()}.jsonl"

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
                        for record in buffer:
                            f.write(json.dumps(record) + '\n')
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

        if not hasattr(ModelRunner.forward, '_sglang_profiled'):
            original = ModelRunner.forward

            def profiled_forward(self, *args, **kwargs):
                start = time.perf_counter()
                batch = args[0] if args else None
                result = original(self, *args, **kwargs)
                duration = time.perf_counter() - start

                tokens = getattr(batch, 'total_num_tokens', 0) or 0
                forward_mode = getattr(batch, 'forward_mode', 1)
                forward_mode_str = 'prefill' if forward_mode == 1 else 'decode'

                seq_lens = getattr(batch, 'extend_seq_lens_cpu', [])

                write_queue.put({
                    "timestamp": start,
                    "process_id": os.getpid(),
                    "batch_size": getattr(batch, 'batch_size', 0),
                    "seq_lens_cpu": seq_lens,
                    "total_num_tokens": tokens,
                    "forward_mode": forward_mode_str,
                    "execution_time_ms": duration * 1000,
                    "tokens_per_second": tokens / duration if duration > 0 and tokens > 0 else 0,
                })
                return result

            profiled_forward._sglang_profiled = True
            ModelRunner.forward = profiled_forward
    except ImportError:
        pass
