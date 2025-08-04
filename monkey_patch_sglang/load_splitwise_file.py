# arrivated_at,prefill,decode
import pandas as pd
import asyncio
import sglang as sgl
import numpy as np
import os

# Set environment variables
os.environ['SGLANG_ENABLE_PROFILING'] = '1'
os.environ['SGLANG_PROFILE_OUTPUT'] = 'model_runner_profile.json'

# Add current directory to PYTHONPATH for subprocesses
current_dir = os.path.dirname(os.path.abspath(__file__))
existing_pythonpath = os.environ.get('PYTHONPATH', '')
if existing_pythonpath:
    os.environ['PYTHONPATH'] = f"{current_dir}:{existing_pythonpath}"
else:
    os.environ['PYTHONPATH'] = current_dir


import working_profiler # This import will trigger the monkey patching

async def data_generator(max_rps, rps_scale=1.0):
    df = pd.read_csv("splitwise_code.csv")
    df["timestamp_filtered"] = df[df["arrived_at"] > 300]["arrived_at"].astype(float)

    df["time_inter_arrival"] = df["timestamp_filtered"].diff().astype(float)
    for index, row in df.iterrows():
        interarrval_time: float = row["time_inter_arrival"]
        interarrval_time = min(interarrval_time, 1.0 / max_rps) if max_rps > 0 else interarrval_time
        await asyncio.sleep(interarrval_time)
        yield {
            "prefill": int(row["num_prefill_tokens"]),
            "decode": int(row["num_decode_tokens"]),
            "time_inter_arrival": row["time_inter_arrival"],
        }


async def launch_jobs(engine, max_time, max_rps, rps_scale=1.0):
    avg_rps = 0.0
    num_jobs = 0
    start_time = asyncio.get_event_loop().time()
    tasks = []
    data_gen = data_generator(max_rps=max_rps, rps_scale=rps_scale)
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
        if elapsed_time >= max_time:
            break
    await asyncio.gather(*tasks)
    print(f"Average RPS: {avg_rps:.2f}, Total Jobs: {num_jobs}")


async def main():
    engine = sgl.Engine(model_path="Qwen/Qwen3-4B")
    # df = pd.read_csv("splitwise_code.csv")
    await asyncio.create_task(launch_jobs(engine, max_time=60, max_rps=10, rps_scale=.5))


if __name__ == "__main__":
    asyncio.run(main())
