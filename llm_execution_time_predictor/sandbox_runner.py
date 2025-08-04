from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional


def run_sandboxed_inference(model_path: str, server_args: Dict[str, Any], backend_name: str,
                           nccl_port: int, batch_size: int, input_len: int, output_len: int,
                           run_name: str, timeout: float = 200.0, chunk_prefill: bool = False,
                           chunk_size: int = 512) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Run inference using sglang_batch_latency.py subprocess with temp file for results"""
    
    # Create temporary file for results
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as result_file:
        result_file_path = result_file.name
    
    try:
        # Build command to run sglang_batch_latency.py directly
        cmd = [
            sys.executable, "-m", "llm_execution_time_predictor.sglang_batch_latency",
            "--model-path", model_path,
            "--tp-size", str(server_args.get('tp_size', 1)),
            "--pp-size", str(server_args.get('pp_size', 1)),
            "--batch-size", str(batch_size),
            "--input-len", str(input_len),
            "--output-len", str(output_len),
            "--run-name", run_name,
            "--result-filename", result_file_path
        ]
        
        # Add chunking parameters if enabled
        if chunk_prefill:
            cmd.extend(["--chunk-prefill"])
            cmd.extend(["--chunk-size", str(chunk_size)])
        
        # Add additional server args
        for key, value in server_args.items():
            if key not in ['tp_size', 'pp_size']:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        # Run subprocess with timeout
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            return None, "timeout"
        
        if process.returncode != 0:
            print(f"[sandbox] Process failed with return code {process.returncode}")
            if stderr:
                print(f"[sandbox] stderr: {stderr.decode()}")
            return None, "oom"
        
        try:
            with open(result_file_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    result_data = json.loads(lines[-1].strip())
                    return result_data, None
                else:
                    return None, "no_result"
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[sandbox] Failed to read result file: {e}")
            return None, "no_result"
            
    finally:
        # Clean up temp file
        try:
            os.unlink(result_file_path)
        except:
            pass