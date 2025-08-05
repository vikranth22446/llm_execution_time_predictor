# SGLang ModelRunner Profiler

Minimal monkey patch for profiling SGLang ModelRunner forward passes.

## Usage

```bash
PYTHONPATH=/path/to/this/folder python simple_patch.py
```

## What it does

- Patches `ModelRunner.forward` to capture timing and batch info
- Saves profiling data to `model_runner_profile.json`
- Runs test with 3 sample prompts

## Output

Captures per forward pass:
- `batch_size` - Number of requests in batch
- `total_num_tokens` - Token count  
- `execution_time_ms` - Forward pass duration
- `forward_mode` - Processing mode
- `process_id` - Subprocess PID