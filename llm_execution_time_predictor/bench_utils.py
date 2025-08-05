from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional
import numpy as np


def powers_of_two(upto: int) -> List[int]:
    out, p = [], 0
    while (1 << p) <= upto:
        out.append(1 << p)
        p += 1
    return out


def hash_key(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()


def extract_gpu_model(gpu_name: str) -> str:
    gpu_name = gpu_name.upper()
    gpu_patterns = [
        r'A100', r'V100', r'H100',
        r'RTX\s*(\d+)', r'GTX\s*(\d+)',
        r'TESLA\s*([A-Z]\d+)', r'QUADRO\s*([A-Z]\d+)',
        r'T4', r'P100', r'K80', r'L4'
    ]
    for pattern in gpu_patterns:
        m = re.search(pattern, gpu_name)
        if m:
            return m.group(0).replace(' ', '')
    m = re.search(r'NVIDIA\s+([A-Z0-9\-]+)', gpu_name)
    if m:
        return m.group(1)
    return ''.join(c for c in gpu_name if c.isalnum())[:10]


def get_gpu_info() -> Dict[str, Any]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        gpu_names = [ln.strip() for ln in result.stdout.strip().splitlines() if ln.strip()]
        if gpu_names:
            from collections import Counter
            counts = Counter(gpu_names)
            return {
                "gpu_type": gpu_names[0],
                "gpu_model": extract_gpu_model(gpu_names[0]),
                "gpu_count": len(gpu_names),
                "gpu_distribution": dict(counts),
            }
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return {"gpu_type": "unknown", "gpu_model": "unknown", "gpu_count": 0, "gpu_distribution": {}}


def average_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results_list:
        return {}
    out = dict(results_list[0])
    numeric_keys = [
        k for k, v in results_list[0].items()
        if isinstance(v, (int, float)) and k not in ("batch_size", "input_len", "output_len")
    ]
    for k in numeric_keys:
        vals = [r[k] for r in results_list if k in r]
        if vals:
            out[k] = float(np.mean(vals))
    for k in ("prefill_latency", "median_decode_latency", "prefill_throughput", "median_decode_throughput"):
        vals = [r.get(k) for r in results_list if r.get(k) is not None]
        if vals:
            out[f"{k}_std"] = float(np.std(vals))
    out["num_runs"] = len(results_list)
    out["runs_data"] = results_list
    return out


def load_json_file(file_path: str, default: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Load JSON file with consistent error handling"""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def save_json_file(file_path: str, data: Dict[str, Any]) -> bool:
    """Save JSON file with consistent error handling"""
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Failed to save {file_path}: {e}")
        return False


def create_backend(backend_name: str):
    """Factory method for backend creation"""
    from .bench_backend_handler import SGLangBackend, VLLMBackend
    return SGLangBackend() if backend_name == "sglang" else VLLMBackend()


def create_rank_printer(tp_rank: int):
    """Create rank-aware print function"""
    return print if tp_rank == 0 else lambda *args, **kwargs: None


def format_output_filename(model_name: str, tp_size: int, backend_name: str, 
                          gpu_model: str, pp_size: Optional[int] = None) -> str:
    """Generate consistent output filename"""
    model_clean = model_name.replace('/', '_')
    if backend_name == "sglang" and pp_size is not None:
        return f"benchmark_data_{model_clean}_TP_{tp_size}_PP_{pp_size}_{gpu_model}.json"
    else:
        return f"benchmark_data_{model_clean}_TP_{tp_size}_{backend_name}_{gpu_model}.json"