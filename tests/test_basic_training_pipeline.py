import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock

sglang_modules = [
    "sglang",
    "sglang.srt",
    "sglang.srt.configs",
    "sglang.srt.configs.model_config",
    "sglang.srt.distributed",
    "sglang.srt.distributed.parallel_state",
    "sglang.srt.entrypoints",
    "sglang.srt.entrypoints.engine",
    "sglang.srt.managers",
    "sglang.srt.managers.schedule_batch",
    "sglang.srt.managers.scheduler",
    "sglang.srt.model_executor",
    "sglang.srt.model_executor.forward_batch_info",
    "sglang.srt.model_executor.model_runner",
    "sglang.srt.sampling",
    "sglang.srt.sampling.sampling_params",
    "sglang.srt.server_args",
    "sglang.srt.speculative",
    "sglang.srt.speculative.spec_info",
    "sglang.srt.utils",
    "sglang.srt.hf_transformers_utils",
]

gpu_deps = ["triton", "torch", "torch.distributed", "transformers"] + sglang_modules
for dep in gpu_deps:
    mock = MagicMock()
    mock.__spec__ = Mock()
    sys.modules[dep] = mock

sys.path.append(str(Path(__file__).parent.parent))

import tempfile
import time

# Import directly from train_utils to avoid __init__.py importing the CLI
from llm_execution_time_predictor.train_utils import (
    get_decode_path_from_model_path, get_model_prefill_path_from_model_path,
    load_onyx_model, predict_lgbm_onyx,
    run_lightgbm_training_pipeline_and_save)


def test_rich_training_pipeline_runs_without_errors():
    """
    Sanity checking loading data from a folder with a lot of jsonl files.
    Using the data to train a model and save it to a onyx format
    """
    model_full_path = Path("profile_output_a100") / Path(
        "deepseek_ai_DeepSeek_R1_Distill_Qwen_1.5B_TP_1"
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        prefill_output_path = Path(temp_dir) / "prefill.onyx"
        decode_output_path = Path(temp_dir) / "decode.onyx"
        _, _, sample_datapoint = run_lightgbm_training_pipeline_and_save(
            model_full_path, prefill_output_path, decode_output_path
        )
        sample_data_prefill, sample_data_decode = sample_datapoint
        model_prefill = load_onyx_model(
            get_model_prefill_path_from_model_path(model_full_path)
        )
        model_decode = load_onyx_model(get_decode_path_from_model_path(model_full_path))

        def warmup():
            for _ in range(5):
                predict_lgbm_onyx(model_prefill, [5], [5], stage="prefill")
                predict_lgbm_onyx(model_decode, [5], [5], stage="decode")

        warmup()
        start_time = time.perf_counter()
        predict_lgbm_onyx(
            model_prefill,
            sample_data_prefill["combined_seq_lens"],
            sample_data_prefill["cached_prefix_lens"],
            stage="prefill",
        )
        end_time = time.perf_counter()

        print(f"Prefill model prediction took {(end_time - start_time) * 1e6:.4f} us")
        start_time = time.perf_counter()
        predict_lgbm_onyx(
            model_decode,
            sample_data_decode["combined_seq_lens"],
            sample_data_decode["cached_prefix_lens"],
            stage="decode",
        )
        end_time = time.perf_counter()
        print(f"Decode model prediction took {(end_time - start_time) * 1e6:.4f} us")
        assert True


if __name__ == "__main__":
    test_rich_training_pipeline_runs_without_errors()
