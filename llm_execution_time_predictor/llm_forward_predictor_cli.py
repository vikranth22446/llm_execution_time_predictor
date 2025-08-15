import argparse
import os
import subprocess
import sys
from pathlib import Path

# Import the existing argument parsing functions
from llm_execution_time_predictor.sglang_batch_latency import (BenchArgs,
                                                               ServerArgs)
from llm_execution_time_predictor.sglang_batch_latency import \
    main as sglang_main
from llm_execution_time_predictor.train_utils import (
    run_lightgbm_training_pipeline_and_save,
    get_model_prefill_path_from_model_path,
    get_decode_path_from_model_path
)

# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent
MONKEY_PATCH_SCRIPT = SCRIPT_DIR / "monkey_patch_sglang" / "run_splitwise_file.py"


def profile_prefill(server_args, bench_args):
    """Profile prefill performance using SGLang batch latency."""
    bench_args.run_prefill_profiling = True
    bench_args.run_decode_profiling = False
    bench_args.correctness_test = False
    sglang_main(server_args, bench_args)


def profile_decode(server_args, bench_args):
    """Profile decode performance using SGLang batch latency."""
    bench_args.run_prefill_profiling = False
    bench_args.run_decode_profiling = True
    bench_args.correctness_test = False
    sglang_main(server_args, bench_args)


def profile_prefill_with_prefix_cache(server_args, bench_args):
    """Profile prefill performance with prefix cache using SGLang batch latency."""
    bench_args.run_prefill_profiling = False
    bench_args.run_decode_profiling = False
    bench_args.run_prefill_profiling_with_prefix_cache = True
    bench_args.correctness_test = False
    sglang_main(server_args, bench_args)


def profile_real(args):
    """Profile real workload using monkey patching script."""
    env = os.environ.copy()

    # Create output directory structure with Model_name_TP_{} pattern
    if args.output_file:
        tp_size = getattr(args, "tp_size", 1)
        model_name = args.model.replace("/", "_").replace("-", "_")
        subfolder_name = f"{model_name}_TP_{tp_size}"

        # Create base directory if output_file contains path
        if "/" in args.output_file:
            base_dir = os.path.dirname(args.output_file)
            filename = os.path.basename(args.output_file)
        else:
            base_dir = "profiling_output"
            filename = args.output_file

        output_path = os.path.join(base_dir, subfolder_name)
        os.makedirs(output_path, exist_ok=True)

        full_output_path = os.path.join(output_path, filename)
        env["SGLANG_PROFILE_OUTPUT"] = full_output_path
        args.output_file = full_output_path  # Update args for command construction

    cmd = [
        sys.executable,
        str(MONKEY_PATCH_SCRIPT),
        "--model",
        args.model,
        "--max_job_send_time",
        str(args.max_job_send_time),
    ]
    if args.data_file:
        cmd.extend(["--data_file", args.data_file])

    if args.max_rps:
        cmd.extend(["--max_rps", str(args.max_rps)])

    if args.rps_scale:
        cmd.extend(["--rps_scale", str(args.rps_scale)])

    if args.output_file:
        cmd.extend(["--output_file", args.output_file])

    if args.max_window_time:
        cmd.extend(["--max_window_time", str(args.max_window_time)])

    if hasattr(args, "tp_size"):
        cmd.extend(["--tp_size", str(args.tp_size)])

    print(f"Running: {' '.join(cmd)}")
    print(
        f"Environment: SGLANG_PROFILE_OUTPUT={env.get('SGLANG_PROFILE_OUTPUT', 'Not set')}"
    )
    return subprocess.run(cmd, env=env)


def train_pipeline(args):
    """Train machine learning models for latency prediction."""
    folder_path = Path(args.folder)
    
    if not folder_path.exists():
        print(f"Error: Folder path {folder_path} does not exist.")
        return 1
    
    # Handle folder vs single model path
    if folder_path.is_file() or any(f.suffix == ".jsonl" for f in folder_path.iterdir() if f.is_file()):
        # Single model path
        model_paths = [folder_path]
    else:
        # Directory containing multiple model paths
        model_paths = [p for p in folder_path.iterdir() if p.is_dir()]
    
    if not model_paths:
        print(f"No valid model paths found in {folder_path}")
        return 1
    
    results = []
    for model_path in model_paths:
        print(f"Training models for {model_path}")
        try:
            # Determine output paths
            if args.prefill_model_output_path:
                prefill_output = Path(args.prefill_model_output_path)
            else:
                prefill_output = get_model_prefill_path_from_model_path(model_path)
            
            if args.decode_model_output_path:
                decode_output = Path(args.decode_model_output_path)
            else:
                decode_output = get_decode_path_from_model_path(model_path)
            
            # Run training pipeline
            _, _, _ = run_lightgbm_training_pipeline_and_save(
                model_path, prefill_output, decode_output
            )
            
            results.append({
                'model_path': str(model_path),
                'prefill_model_path': str(prefill_output),
                'decode_model_path': str(decode_output)
            })
            print(f"Successfully trained models for {model_path}")
            print(f"  Prefill model: {prefill_output}")
            print(f"  Decode model: {decode_output}")
            
        except Exception as e:
            print(f"Error training models for {model_path}: {e}")
            continue
    
    # Print summary
    print(f"\nTrained models for {len(results)} out of {len(model_paths)} model paths:")
    for result in results:
        print(f"Model: {result['model_path']}")
        print(f"  Prefill: {result['prefill_model_path']}")
        print(f"  Decode: {result['decode_model_path']}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="LLM Forward Predictor CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Profile prefill performance
            %(prog)s profile prefill --model-path meta-llama/Meta-Llama-3-8B-Instruct

            # Profile decode performance  
            %(prog)s profile decode --model-path meta-llama/Meta-Llama-3-8B-Instruct --max-decode-token-length 8192

            # Profile prefill performance with prefix cache
            %(prog)s profile prefill-prefix-cache --model-path meta-llama/Meta-Llama-3-8B-Instruct

            # Profile real workload
            %(prog)s profile_real --model Qwen/Qwen3-8B --output_file qwen3_8b_log.jsonl --max_job_send_time 10
            
            # Train latency prediction models
            %(prog)s train --folder /path/to/model/profile/data
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Profile command with subcommands
    profile_parser = subparsers.add_parser("profile", help="Profile performance")
    profile_subparsers = profile_parser.add_subparsers(
        dest="profile_type", help="Profile type"
    )

    # Prefill profiling - reuse existing argument parsers
    prefill_parser = profile_subparsers.add_parser(
        "prefill", help="Profile prefill performance"
    )
    ServerArgs.add_cli_args(prefill_parser)
    BenchArgs.add_cli_args(prefill_parser)

    # Decode profiling - reuse existing argument parsers
    decode_parser = profile_subparsers.add_parser(
        "decode", help="Profile decode performance"
    )
    ServerArgs.add_cli_args(decode_parser)
    BenchArgs.add_cli_args(decode_parser)

    # Prefill profiling with prefix cache - reuse existing argument parsers
    prefill_prefix_cache_parser = profile_subparsers.add_parser(
        "prefill-prefix-cache", help="Profile prefill performance with prefix cache"
    )
    ServerArgs.add_cli_args(prefill_prefix_cache_parser)
    BenchArgs.add_cli_args(prefill_prefix_cache_parser)

    # Real workload profiling
    real_parser = subparsers.add_parser(
        "profile_real", help="Profile real workload using monkey patching"
    )
    real_parser.add_argument("--model", type=str, required=True, help="Model name/path")
    real_parser.add_argument(
        "--output_file", type=str, help="Output file for profiling results"
    )
    real_parser.add_argument(
        "--max_job_send_time",
        type=int,
        default=60,
        help="Maximum job send time in seconds",
    )
    real_parser.add_argument(
        "--data_file",
        type=str,
        default="data/splitwise_code.csv",
        help="Data file to use",
    )
    real_parser.add_argument(
        "--max_rps", type=int, default=10, help="Maximum requests per second"
    )
    real_parser.add_argument(
        "--rps_scale", type=float, default=1.0, help="RPS scaling factor"
    )
    real_parser.add_argument(
        "--max_window_time",
        type=int,
        default=300,
        help="Maximum window time in seconds",
    )
    real_parser.add_argument(
        "--tp_size", type=int, default=1, help="Tensor parallelism size"
    )
    
    # Train command
    train_parser = subparsers.add_parser(
        "train", help="Train latency prediction models"
    )
    train_parser.add_argument(
        "--folder", 
        type=str, 
        required=True, 
        help="Path to folder containing model profile data or single model path"
    )
    train_parser.add_argument(
        "--prefill-model-output-path",
        type=str,
        help="Custom output path for prefill model (default: <model_path>/prefill_model.onnx)"
    )
    train_parser.add_argument(
        "--decode-model-output-path", 
        type=str,
        help="Custom output path for decode model (default: <model_path>/decode_model.onnx)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Validate script paths exist for profile_real
    if args.command == "profile_real" and not MONKEY_PATCH_SCRIPT.exists():
        print(f"Error: Monkey patch script not found at {MONKEY_PATCH_SCRIPT}")
        return 1

    # Execute the appropriate command
    if args.command == "profile":
        if args.profile_type == "prefill":
            server_args = ServerArgs.from_cli_args(args)
            bench_args = BenchArgs.from_cli_args(args)
            profile_prefill(server_args, bench_args)
        elif args.profile_type == "decode":
            server_args = ServerArgs.from_cli_args(args)
            bench_args = BenchArgs.from_cli_args(args)
            profile_decode(server_args, bench_args)
        elif args.profile_type == "prefill-prefix-cache":
            server_args = ServerArgs.from_cli_args(args)
            bench_args = BenchArgs.from_cli_args(args)
            profile_prefill_with_prefix_cache(server_args, bench_args)
        else:
            profile_parser.print_help()
            return 1
    elif args.command == "profile_real":
        result = profile_real(args)
        return result.returncode if result else 0
    elif args.command == "train":
        return train_pipeline(args)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    main()
