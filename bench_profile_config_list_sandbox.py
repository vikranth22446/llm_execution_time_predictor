#!/usr/bin/env python3
import os, shlex, subprocess, threading

JOBS = [
    ("Qwen/Qwen3-4B",  [1]),
    ("Qwen/Qwen3-8B",  [1, 2, 4, 8]),
    ("Qwen/Qwen3-14B", [2]),
    ("Qwen/Qwen3-32B", [2, 4]),
    ("meta-llama/Llama-3.1-8B", [1]),
]

CMD = ("python3 llm_execution_time_predictor/llm_forward_predictor_cli.py "
       "profile {model} --tp_size {tp} --sandbox")

TASKS = [(m, tp) for m, tps in JOBS for tp in tps]
LOCK = threading.Lock()

def pop_best_task(max_size: int):
    with LOCK:
        best_idx = -1
        best_size = 0
        for i, (m, tp) in enumerate(TASKS):
            if tp <= max_size and tp > best_size:
                best_idx = i
                best_size = tp
        if best_idx >= 0:
            return TASKS.pop(best_idx)
    return None

def run_one(devs, model, tp):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devs))
    cmd = CMD.format(model=model, tp=tp)
    print(f"[GPUs {devs}] {model} tp={tp}")
    subprocess.run(shlex.split(cmd), env=env)

def schedule(devs):
    available = set(devs)
    running = []  # (thread, used_gpus)
    
    while TASKS or running:
        # Check for completed threads and release their GPUs
        for i in range(len(running) - 1, -1, -1):
            thread, used_gpus = running[i]
            if not thread.is_alive():
                available.update(used_gpus)
                running.pop(i)
        
        # Try to start new tasks with available GPUs
        if TASKS and len(available) > 0:
            task = pop_best_task(len(available))
            if task:
                model, tp = task
                used_gpus = list(available)[:tp]
                available -= set(used_gpus)
                
                t = threading.Thread(target=run_one, args=(used_gpus, model, tp))
                t.start()
                running.append((t, used_gpus))
        
        # Small sleep to avoid busy waiting
        if running:
            import time
            time.sleep(0.1)

def main():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devs = [int(x) for x in vis.split(",") if x.strip()] or [0]
    schedule(devs)
    if TASKS:
        print("Unscheduled (no matching GPU group size):", TASKS)

if __name__ == "__main__":
    main()