from tqdm.asyncio import tqdm_asyncio# arrivated_at,prefill,decode
import pandas as pd
import asyncio
import sglang as sgl
import numpy as np
import os
import tempfile
import random
import uuid

random_filename = str(uuid.uuid4())
random_filepath = f"{random_filename}_model_runner_profile.jsonl"
# Set environment variables
os.environ['SGLANG_ENABLE_PROFILING'] = '1'

if not os.environ["SGLANG_PROFILE_OUTPUT"]:
    os.environ['SGLANG_PROFILE_OUTPUT'] = random_filepath

# Add current directory to PYTHONPATH for subprocesses
current_dir = os.path.dirname(os.path.abspath(__file__))
existing_pythonpath = os.environ.get('PYTHONPATH', '')
if existing_pythonpath:
    os.environ['PYTHONPATH'] = f"{current_dir}:{existing_pythonpath}"
else:
    os.environ['PYTHONPATH'] = current_dir


import working_profiler # This import will trigger the monkey patching

async def data_generator(data_file, max_rps, rps_scale=1.0):
    df = pd.read_csv(data_file)
    # Filter the start of the dataset
    df["timestamp_filtered"] = df[df["arrived_at"] > 300]["arrived_at"].astype(float)

    df["time_inter_arrival"] = df["timestamp_filtered"].diff().astype(float)
    for index, row in df.iterrows():
        interarrval_time: float = row["time_inter_arrival"]
        interarrval_time = max(interarrval_time, 1.0 / max_rps) if max_rps > 0 else interarrval_time
        await asyncio.sleep(interarrval_time)
        yield {
            "prefill": int(row["num_prefill_tokens"]),
            "decode": int(row["num_decode_tokens"]),
            "time_inter_arrival": row["time_inter_arrival"],
        }


async def launch_jobs(engine,data_file, max_job_send_time, max_rps, rps_scale=1.0):
    avg_rps = 0.0
    num_jobs = 0
    start_time = asyncio.get_event_loop().time()
    tasks = []
    data_gen = data_generator(data_file, max_rps=max_rps, rps_scale=rps_scale)
    async for data in data_gen:
        prefill = data["prefill"]
        decode = data["decode"]
        num_jobs += 1
        current_time = asyncio.get_event_loop().time()
        elapsed_time = current_time - start_time
        avg_rps = num_jobs / elapsed_time if elapsed_time > 0 else 0
        prompt_ids = list(np.random.randint(0, 100000, size=prefill))
        tasks.append(
            engine.async_generate(
                input_ids=[prompt_ids],
                sampling_params={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_new_tokens": decode,
                    "ignore_eos": True,
                },
            )
        )
        if elapsed_time >= max_job_send_time:
            break
    await tqdm_asyncio.gather(*tasks)
    print(f"Average RPS: {avg_rps:.2f}, Total Jobs: {num_jobs}")
    # Move The Temp file to custom dir

async def main(model,data_file, max_job_send_time, max_rps, rps_scale):
    engine = sgl.Engine(model_path=model)
    await asyncio.create_task(launch_jobs(engine,data_file, max_job_send_time=max_job_send_time, max_rps=max_rps, rps_scale=rps_scale))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--max_job_send_time", type=int, default=60)
    parser.add_argument("--max_rps", type=int, default=10)
    parser.add_argument("--rps_scale", type=float, default=1.0)
    parser.add_argument("--data_file", type=str, default="data/splitwise_code.csv")
    parser.add_argument("--output_file", type=str, default=random_filepath)
    args = parser.parse_args()
    asyncio.run(main(args.model, args.data_file, args.max_job_send_time, args.max_rps, args.rps_scale))
